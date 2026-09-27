"""Data-fitted, non-destructive business-record normalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from difflib import SequenceMatcher
from itertools import chain, islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

ARTIFACT_VERSION = 1
FIELD_ALIASES = {
    "id": {"id", "entityid", "recordid", "businessid"},
    "name": {"name", "businessname", "companyname", "organisationname", "organizationname"},
    "address": {"address", "businessaddress", "streetaddress", "location", "fulladdress"},
    "city": {"city", "town", "locality"},
    "postal": {"postal", "postalcode", "postcode", "zipcode", "zip", "pincode", "pin"},
    "country": {"country", "countryname", "nation"},
}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[./-]\d+)*", re.UNICODE)


def _column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _clean_text(value: Any, strip_diacritics: bool = True) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    if strip_diacritics:
        text = "".join(
            char for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    return " ".join(TOKEN_RE.findall(text))


def _tokens(value: Any, strip_diacritics: bool = True) -> list[str]:
    return TOKEN_RE.findall(_clean_text(value, strip_diacritics))


def _iter_records(data: Any) -> Iterator[dict[str, Any]]:
    if hasattr(data, "to_dict") and data.__class__.__module__.startswith("pandas"):
        yield from data.to_dict(orient="records")
        return
    for row in data:
        if not isinstance(row, Mapping):
            raise TypeError("Normalizer input must contain mapping-like records")
        yield dict(row)


def _field_scores(columns: Sequence[Any]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for column in columns:
        key = _column_key(column)
        scores[str(column)] = {}
        for field, aliases in FIELD_ALIASES.items():
            score = 1.0 if key in aliases else 0.0
            if not score:
                score = max(
                    (0.85 for alias in aliases if len(alias) > 3 and alias in key),
                    default=0.0,
                )
            scores[str(column)][field] = score
    return scores


class Normalizer:
    """Learn field vocabularies from records, then produce derived features."""

    def __init__(self, strip_diacritics: bool = True) -> None:
        self.strip_diacritics = strip_diacritics
        self.schema: dict[str, str] = {}
        self.token_frequency: Counter[str] = Counter()
        self.document_frequency: Counter[str] = Counter()
        self.contexts: dict[str, Counter[str]] = defaultdict(Counter)
        self.city_frequency: Counter[str] = Counter()
        self.locality_frequency: Counter[str] = Counter()
        self.terminal_frequency: Counter[str] = Counter()
        self.street_adjacency: Counter[str] = Counter()
        self.postal_patterns: Counter[str] = Counter()
        self.row_count = 0
        self._hasher = hashlib.sha256()
        self._prior_sha256: str | None = None
        self._fitted = False
        self.abbreviations: dict[str, str] = {}
        self.legal_suffixes: list[str] = []
        self.street_words: list[str] = []
        self._street_word_set: set[str] = set()
        self.postal_pattern: dict[str, Any] | None = None

    @staticmethod
    def _sample(data: Any, limit: int = 256) -> tuple[list[dict[str, Any]], Iterator[dict[str, Any]]]:
        records = _iter_records(data)
        sample = list(islice(records, limit))
        return sample, chain(sample, records)

    def _detect_schema(self, sample: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        if not sample:
            raise ValueError("Cannot fit a normalizer from zero records")
        columns = list(dict.fromkeys(key for row in sample for key in row))
        scores = _field_scores(columns)
        chosen: dict[str, str] = {}
        used: set[str] = set()
        for field in FIELD_ALIASES:
            options = sorted(
                ((scores[str(column)][field], str(column)) for column in columns),
                reverse=True,
            )
            if options and options[0][0] > 0:
                best_score, best_column = options[0]
                if best_column not in used and (len(options) == 1 or best_score > options[1][0]):
                    chosen[field] = best_column
                    used.add(best_column)

        for field, predicate in (
            ("postal", lambda value: bool(re.fullmatch(r"[\da-zA-Z -]{3,12}", value.strip()))),
            ("name", lambda value: len(_tokens(value, self.strip_diacritics)) > 0 and not value.strip().isdigit()),
            ("address", lambda value: len(_tokens(value, self.strip_diacritics)) >= 2),
        ):
            if field in chosen:
                continue
            candidates = []
            for column in columns:
                if column in used:
                    continue
                values = [str(row[column]) for row in sample if row.get(column) not in (None, "")]
                if not values:
                    continue
                ratio = sum(predicate(value) for value in values) / len(values)
                if field == "postal":
                    ratio = sum(bool(re.fullmatch(r"\d{4,10}|(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9 -]{4,10}", v.strip())) for v in values) / len(values)
                    if ratio < 0.8:
                        continue
                candidates.append((ratio, str(column)))
            if candidates:
                candidates.sort(reverse=True)
                ratio, column = candidates[0]
                unambiguous = len(candidates) == 1 or ratio - candidates[1][0] >= 0.1
                if ratio >= (0.8 if field == "postal" else 0.6) and unambiguous:
                    chosen[field] = column
                    used.add(column)
        return chosen

    def _refresh_learned_tables(self) -> None:
        context_tokens: dict[str, set[str]] = defaultdict(set)
        for token, contexts in self.contexts.items():
            for context in contexts:
                if len(context_tokens[context]) <= 128:
                    context_tokens[context].add(token)

        shared_contexts: Counter[tuple[str, str]] = Counter()
        for neighboring_tokens in context_tokens.values():
            if len(neighboring_tokens) > 128:
                continue
            short_tokens = [token for token in neighboring_tokens if 2 <= len(token) <= 6]
            long_tokens = [token for token in neighboring_tokens if 2 <= len(token) <= 20]
            for short in short_tokens:
                for long in long_tokens:
                    if len(long) >= len(short) + 2 and short != long:
                        shared_contexts[(short, long)] += 1

        abbreviation_candidates: dict[str, tuple[str, float]] = {}
        for (short, long), shared_count in shared_contexts.items():
            if self.token_frequency[short] < 2 or self.token_frequency[long] < 2 or shared_count < 1:
                continue
            short_contexts = self.contexts[short]
            long_contexts = self.contexts[long]
            shared = sum((short_contexts & long_contexts).values())
            union = sum((short_contexts | long_contexts).values())
            overlap = shared / union if union else 0.0
            similarity = SequenceMatcher(None, short, long).ratio()
            remaining = iter(long)
            is_subsequence = len(short) >= 3 and all(
                any(char == candidate for candidate in remaining) for char in short
            )
            if overlap >= 0.2 and (similarity >= 0.35 or is_subsequence):
                current = abbreviation_candidates.get(short)
                if current is None or overlap > current[1]:
                    abbreviation_candidates[short] = (long, overlap)
        self.abbreviations = {short: value[0] for short, value in abbreviation_candidates.items()}

        self.legal_suffixes = sorted(
            token for token, count in self.terminal_frequency.items()
            if count >= 3 and count / max(self.token_frequency[token], 1) >= 0.6
        )
        self.street_words = sorted(
            token for token, count in self.street_adjacency.items()
            if count >= 2 and count / max(self.token_frequency[token], 1) >= 0.1
        )
        self._street_word_set = set(self.street_words)
        if self.postal_patterns:
            signature, count = self.postal_patterns.most_common(1)[0]
            length, kind = signature.split(":", 1)
            self.postal_pattern = {
                "length": int(length),
                "kind": kind,
                "support": count,
                "share": count / sum(self.postal_patterns.values()),
            }
        else:
            self.postal_pattern = None

    def _observe(self, record: Mapping[str, Any]) -> None:
        schema = self.schema
        name = record.get(schema.get("name", ""), "")
        address = record.get(schema.get("address", ""), "")
        values = [name, address]
        if schema.get("city"):
            values.append(record.get(schema["city"], ""))
        record_tokens: set[str] = set()
        for value in values:
            tokens = _tokens(value, self.strip_diacritics)
            self.token_frequency.update(tokens)
            record_tokens.update(tokens)
            for index, token in enumerate(tokens):
                contexts = self.contexts[token]
                neighbors = []
                if index:
                    neighbors.append(f"L:{tokens[index - 1]}")
                if index + 1 < len(tokens):
                    neighbors.append(f"R:{tokens[index + 1]}")
                for context in neighbors:
                    if context in contexts or len(contexts) < 32:
                        contexts[context] += 1
        self.document_frequency.update(record_tokens)

        if name_tokens := _tokens(name, self.strip_diacritics):
            self.terminal_frequency[name_tokens[-1]] += 1
        if schema.get("city"):
            city = _clean_text(record.get(schema["city"], ""), self.strip_diacritics)
            if city:
                self.city_frequency[city] += 1

        address_tokens = _tokens(address, self.strip_diacritics)
        if address_tokens and not any(char.isdigit() for char in address_tokens[-1]):
            self.locality_frequency[address_tokens[-1]] += 1
        for index, token in enumerate(address_tokens):
            previous_number = index > 0 and any(char.isdigit() for char in address_tokens[index - 1])
            next_number = index + 1 < len(address_tokens) and any(char.isdigit() for char in address_tokens[index + 1])
            if previous_number or next_number:
                self.street_adjacency[token] += 1

        postal_column = schema.get("postal")
        postal_values = [record.get(postal_column, "")] if postal_column else NUMBER_RE.findall(str(address or ""))
        for value in postal_values:
            match = re.search(r"[A-Za-z0-9]+", str(value))
            if not match:
                continue
            code = match.group(0)
            if not postal_column and len(code) < 4:
                continue
            kind = "digits" if code.isdigit() else "alphanumeric"
            self.postal_patterns[f"{len(code)}:{kind}"] += 1

        self.row_count += 1
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        if self._prior_sha256 is not None:
            self._hasher.update(self._prior_sha256.encode("ascii"))
            self._hasher.update(b"\n")
            self._prior_sha256 = None
        self._hasher.update(encoded.encode("utf-8"))
        self._hasher.update(b"\n")

    def fit(self, data: Any, schema: Mapping[str, str] | None = None) -> "Normalizer":
        self.__init__(strip_diacritics=self.strip_diacritics)
        sample, records = self._sample(data)
        self.schema = self._detect_schema(sample)
        if schema:
            unknown_fields = set(schema) - set(FIELD_ALIASES)
            if unknown_fields:
                raise ValueError(f"Unknown schema fields: {sorted(unknown_fields)}")
            available = {key for row in sample for key in row}
            missing_columns = set(schema.values()) - available
            if missing_columns:
                raise ValueError(f"Schema columns not found in input: {sorted(missing_columns)}")
            self.schema.update(schema)
        for record in records:
            self._observe(record)
        self._fitted = True
        self._refresh_learned_tables()
        return self

    def partial_fit(self, data: Any) -> "Normalizer":
        sample, records = self._sample(data)
        if not sample:
            return self
        if not self._fitted:
            self.schema = self._detect_schema(sample)
        else:
            for field, column in self.schema.items():
                if any(column not in row for row in sample):
                    raise ValueError(f"Incoming records do not contain fitted {field} column {column!r}")
        for record in records:
            self._observe(record)
        self._fitted = True
        self._refresh_learned_tables()
        return self

    def transform(self, data: Any) -> Any:
        if not self._fitted:
            raise RuntimeError("Call fit() or load an artifact before transform()")
        is_frame = hasattr(data, "to_dict") and data.__class__.__module__.startswith("pandas")
        source_index = data.index if is_frame else None
        output = list(self.iter_transform(data))
        if is_frame:
            import pandas as pd
            return pd.DataFrame(output, index=source_index)
        return output

    def iter_transform(self, data: Any) -> Iterator[dict[str, Any]]:
        if not self._fitted:
            raise RuntimeError("Call fit() or load an artifact before transform()")
        for record in _iter_records(data):
            yield self._transform_record(record)

    def _transform_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(record)
        schema = self.schema
        raw_name = record.get(schema.get("name", ""), "")
        raw_address = record.get(schema.get("address", ""), "")
        name_tokens = _tokens(raw_name, self.strip_diacritics)
        address_tokens = _tokens(raw_address, self.strip_diacritics)
        all_tokens = name_tokens + address_tokens
        normalized_name_tokens = [self.abbreviations.get(token, token) for token in name_tokens]
        normalized_address_tokens = [self.abbreviations.get(token, token) for token in address_tokens]
        normalized_address = " ".join(normalized_address_tokens)
        raw_numbers = NUMBER_RE.findall(f"{raw_name} {raw_address}")

        components: dict[str, Any] = {}
        confidence: dict[str, str] = {}
        if raw_numbers:
            components["numbers"] = raw_numbers
            confidence["numbers"] = "low" if len(raw_numbers) > 1 else "medium"
        postal_value = record.get(schema.get("postal", ""), "") if schema.get("postal") else ""
        postal_candidates = [str(postal_value)] if postal_value else []
        if self.postal_pattern:
            postal_candidates.extend(
                token for token in address_tokens
                if len(token) == self.postal_pattern["length"]
                and (token.isdigit() if self.postal_pattern["kind"] == "digits" else token.isalnum())
            )
        if postal_candidates:
            components["postal_code"] = postal_candidates[0]
            confidence["postal_code"] = "medium" if schema.get("postal") else "low"
        if address_tokens and address_tokens[0].isdigit():
            components["house_no_candidates"] = [address_tokens[0]]
            postal_like = self.postal_pattern and len(address_tokens[0]) == self.postal_pattern["length"]
            confidence["house_no_candidates"] = "low" if len(raw_numbers) > 1 or postal_like else "medium"

        street_hits = [token for token in address_tokens if token in self._street_word_set]
        if street_hits:
            components["street_indicators"] = street_hits
            confidence["street_indicators"] = "medium"
        if schema.get("country") and record.get(schema["country"]) not in (None, ""):
            components["country"] = record[schema["country"]]
            confidence["country"] = "high"
        if schema.get("city") and record.get(schema["city"]) not in (None, ""):
            components["city"] = record[schema["city"]]
            confidence["city"] = "high"
        locality_hits = [token for token in address_tokens[-2:] if self.locality_frequency[token] >= 3]
        if locality_hits:
            components["locality_candidates"] = locality_hits
            confidence["locality_candidates"] = "low"

        document_total = max(self.row_count, 1)
        weights = {
            token: round(math.log1p(document_total / (1 + self.document_frequency.get(token, 0))), 6)
            for token in dict.fromkeys(all_tokens)
        }
        result.update({
            "raw_name": raw_name,
            "raw_address": raw_address,
            "normalized_name": " ".join(normalized_name_tokens),
            "normalized_address": normalized_address,
            "tokens": all_tokens,
            "components": components,
            "numeric_features": {"numbers": raw_numbers, "number_count": len(raw_numbers)},
            "token_weights": weights,
            "component_confidence": confidence,
        })
        return result

    def fit_transform(self, data: Any, schema: Mapping[str, str] | None = None) -> Any:
        records = list(_iter_records(data))
        return self.fit(records, schema=schema).transform(records)

    def validate(self, data: Any, max_empty_component_ratio: float = 0.1, max_unknown_token_ratio: float = 0.5) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Call fit() or load an artifact before validate()")
        row_count = empty_count = token_count = unknown_count = postal_matches = 0
        postal_observations = 0
        for row in _iter_records(data):
            transformed = self._transform_record(row)
            row_count += 1
            if not transformed["components"]:
                empty_count += 1
            tokens = transformed["tokens"]
            token_count += len(tokens)
            unknown_count += sum(token not in self.token_frequency for token in tokens)
            if self.postal_pattern:
                postal_observations += 1
                postal_matches += "postal_code" in transformed["components"]
        empty_ratio = empty_count / max(row_count, 1)
        unknown_ratio = unknown_count / max(token_count, 1)
        checks = {
            "empty_component_ratio": {"value": empty_ratio, "limit": max_empty_component_ratio, "passed": empty_ratio <= max_empty_component_ratio},
            "unknown_token_ratio": {"value": unknown_ratio, "limit": max_unknown_token_ratio, "passed": unknown_ratio <= max_unknown_token_ratio},
        }
        if self.postal_pattern:
            coverage = postal_matches / max(postal_observations, 1)
            checks["postal_pattern_coverage"] = {"value": coverage, "limit": 0.05, "passed": coverage >= 0.05}
        return {
            "row_count": row_count,
            "checks": checks,
            "passed": all(check["passed"] for check in checks.values()),
            "schema": dict(self.schema),
        }

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted normalizer")
        artifact = {
            "version": ARTIFACT_VERSION,
            "strip_diacritics": self.strip_diacritics,
            "schema": self.schema,
            "row_count": self.row_count,
            "dataset_sha256": self.dataset_sha256,
            "token_frequency": dict(self.token_frequency),
            "document_frequency": dict(self.document_frequency),
            "contexts": {key: dict(value) for key, value in self.contexts.items()},
            "city_frequency": dict(self.city_frequency),
            "locality_frequency": dict(self.locality_frequency),
            "terminal_frequency": dict(self.terminal_frequency),
            "street_adjacency": dict(self.street_adjacency),
            "postal_patterns": dict(self.postal_patterns),
            "abbreviations": self.abbreviations,
            "legal_suffixes": self.legal_suffixes,
            "street_words": self.street_words,
            "postal_pattern": self.postal_pattern,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if artifact.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported normalizer artifact version: {artifact.get('version')!r}")
        normalizer = cls(strip_diacritics=artifact["strip_diacritics"])
        normalizer.schema = artifact["schema"]
        normalizer.row_count = artifact["row_count"]
        normalizer.token_frequency = Counter(artifact["token_frequency"])
        normalizer.document_frequency = Counter(artifact["document_frequency"])
        normalizer.contexts = defaultdict(Counter, {key: Counter(value) for key, value in artifact["contexts"].items()})
        normalizer.city_frequency = Counter(artifact.get("city_frequency", {}))
        normalizer.locality_frequency = Counter(artifact.get("locality_frequency", {}))
        normalizer.terminal_frequency = Counter(artifact["terminal_frequency"])
        normalizer.street_adjacency = Counter(artifact["street_adjacency"])
        normalizer.postal_patterns = Counter(artifact["postal_patterns"])
        normalizer.abbreviations = artifact["abbreviations"]
        normalizer.legal_suffixes = artifact["legal_suffixes"]
        normalizer.street_words = artifact["street_words"]
        normalizer._street_word_set = set(normalizer.street_words)
        normalizer.postal_pattern = artifact["postal_pattern"]
        normalizer._hasher = hashlib.sha256()
        normalizer._prior_sha256 = artifact["dataset_sha256"]
        normalizer._fitted = True
        return normalizer

    @property
    def dataset_sha256(self) -> str:
        return self._prior_sha256 or self._hasher.hexdigest()


@contextmanager
def open_tsv(path: str | Path, member: str | None = None) -> Iterator[TextIO]:
    """Open a TSV on disk or a TSV member inside a ZIP archive."""
    source = Path(path)
    if zipfile.is_zipfile(source):
        if member is None:
            raise ValueError("A ZIP input requires --member")
        with zipfile.ZipFile(source) as archive:
            with archive.open(member) as binary:
                import io
                yield io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    else:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            yield stream


def read_tsv_records(path: str | Path, member: str | None = None) -> Iterator[dict[str, str]]:
    with open_tsv(path, member) as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def write_tsv_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    iterator = iter(records)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Cannot write TSV without records") from None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(first)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in chain((first,), iterator):
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="Fit from one or more TSV files or ZIP members")
    fit_parser.add_argument("--input", required=True, help="TSV file or ZIP archive")
    fit_parser.add_argument("--member", action="append", help="ZIP member (repeat for multiple datasets)")
    fit_parser.add_argument("--artifact", required=True)
    transform_parser = subparsers.add_parser("transform", help="Transform a TSV using a saved artifact")
    transform_parser.add_argument("--input", required=True)
    transform_parser.add_argument("--member", help="ZIP member when input is an archive")
    transform_parser.add_argument("--artifact", required=True)
    transform_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        members = args.member or [None]
        normalizer = Normalizer()
        for member in members:
            normalizer.partial_fit(read_tsv_records(args.input, member))
        normalizer.save(args.artifact)
        print(json.dumps({"row_count": normalizer.row_count, "dataset_sha256": normalizer.dataset_sha256, "schema": normalizer.schema}, indent=2))
    else:
        normalizer = Normalizer.load(args.artifact)
        records = normalizer.iter_transform(read_tsv_records(args.input, args.member))
        write_tsv_records(args.output, records)
        print(json.dumps(normalizer.validate(read_tsv_records(args.input, args.member)), indent=2))


if __name__ == "__main__":
    _cli()
