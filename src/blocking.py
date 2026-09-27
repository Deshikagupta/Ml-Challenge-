"""Multi-pass blocking / candidate-generation layer.

This module sits **after** the existing, frozen ``Normalizer`` and **before**
the future matching model:

    RAW DATA -> Normalizer -> normalized records -> blocking (this module) ->
    candidate_pairs.tsv -> (future) matching model

It never re-normalizes, re-tokenizes, or re-derives anything the Normalizer
already computed. Every blocking pass reads directly from the fields the
Normalizer produces on each record:

    normalized_name, normalized_address, components, component_confidence

Five blocking passes are implemented, each a separate function/class so they
can be measured independently:

    Block A  exact_name_candidates       inverted index on normalized_name
    Block B  token_candidates            rare/informative-token inverted index
    Block C  address_candidates          multiple component-based key indexes
    Block D  tfidf_name_candidates /     sparse char n-gram TF-IDF retrieval
             tfidf_address_candidates
    Block E  country_token_candidates    country-partitioned token index
             (additive only -- never used to eliminate a candidate found by
             another block; a missing/unseen country simply contributes
             nothing extra)

All indexes are built with a **single streaming pass** per source (Source 2,
Source 3 are always indexed separately -- see ``BlockingIndex``) and every
posting list is capped so no single high-frequency key can blow up recall
into a huge candidate list. Candidate lookups never do an S1 x S2/S3 nested
Python loop: Blocks A/B/C/E are O(1) dict lookups per query token/key, and
Block D uses a single sparse-matrix multiply per query batch.

See ``CandidateGenerator`` for how the five passes are unioned per S1 record,
and ``evaluate_blocking`` for the training-data recall/volume evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from normalizer import read_tsv_records, write_tsv_records  # reuse existing infra, do not reimplement

# sklearn is an already-common dependency for retrieval tasks like this one; it is only
# imported lazily inside TfidfCharIndex so that exact/token/address blocking (Blocks A/B/C/E)
# work even in an environment where scikit-learn is not installed.


# ======================================================================================
# Configuration
# ======================================================================================

DEFAULT_ENABLED_BLOCKS = frozenset(
    {"exact_name", "token", "address", "tfidf_name", "tfidf_address", "country_token"}
)


@dataclass(frozen=True)
class BlockingConfig:
    """All tunable knobs for every blocking pass, in one place.

    Nothing here is a fixed constant baked into the algorithm -- every value
    is expected to be swept (e.g. tfidf_name_top_k in {10, 20, 30, 50}) when
    tuning the recall/volume tradeoff (see requirement #12).
    """

    id_field: str = "entity_id"
    enabled_blocks: frozenset = DEFAULT_ENABLED_BLOCKS

    # Block B -- name token / partial-name blocking
    token_min_length: int = 3
    token_max_posting_list: int = 2000       # a token posting list longer than this is dropped as a key
    token_max_per_record: int = 6            # how many (rarest-first) tokens a query record uses
    token_use_rare_only: bool = True         # rank candidate keys by rarity before selecting them

    # Block C -- address / component blocking
    address_key_max_posting_list: int = 5000

    # Block D -- TF-IDF character n-gram retrieval
    tfidf_name_top_k: int = 30
    tfidf_address_top_k: int = 30
    tfidf_ngram_range: Tuple[int, int] = (2, 4)
    tfidf_max_features: int = 50_000
    tfidf_min_df: int = 1
    tfidf_batch_size: int = 512
    tfidf_min_similarity: float = 0.05

    # Block E -- country-aware restriction/indexing
    country_token_max_posting_list: int = 2000
    country_token_max_per_record: int = 6

    def with_overrides(self, **overrides: Any) -> "BlockingConfig":
        return replace(self, **overrides)


# ======================================================================================
# Small, shared helpers for reading Normalizer output safely (never re-derives fields)
# ======================================================================================

def _key_norm(value: Any) -> Optional[str]:
    """Minimal, non-semantic canonicalization for dict-key matching only.

    This does NOT re-normalize entity content -- it only makes two already-normalized
    values compare equal as dict keys (e.g. trims incidental whitespace). Required by
    requirement #1's "minimal preprocessing technically required by a retrieval key".
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _component(record: Mapping[str, Any], key: str) -> Any:
    components = record.get("components")
    if not isinstance(components, dict):
        return None
    return components.get(key)


def _confidence(record: Mapping[str, Any], key: str) -> Optional[str]:
    conf = record.get("component_confidence")
    if not isinstance(conf, dict):
        return None
    return conf.get(key)


def _first(value: Any) -> Any:
    """components values are sometimes a list (e.g. house_no_candidates); take the first."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _name_tokens(record: Mapping[str, Any], min_length: int) -> List[str]:
    """Tokens straight from the Normalizer's own normalized_name -- no re-tokenizing."""
    normalized_name = record.get("normalized_name") or ""
    return [tok for tok in normalized_name.split() if len(tok) >= min_length]


def _record_id(record: Mapping[str, Any], id_field: str) -> str:
    value = record.get(id_field)
    if value in (None, ""):
        raise KeyError(
            f"Record is missing id field {id_field!r}; pass --id-field to match your data "
            f"(available keys: {sorted(record.keys())[:10]}...)"
        )
    return str(value)


# ======================================================================================
# Generic capped posting-list index (used by Blocks A, B, C, E)
# ======================================================================================

class _PostingIndex:
    """key -> [ids], single streaming pass, with a hard cap on posting-list size.

    Once a key's posting list would exceed ``max_posting_list``, the key is banned:
    its list is dropped and no further ids are added under it. This bounds memory and
    is exactly the "ignore as a blocking key" strategy required for high-frequency
    tokens/keys (requirement #5), applied without ever materializing frequency counts
    up front (single pass, streaming-friendly for very large sources).
    """

    __slots__ = ("max_posting_list", "_postings", "_banned")

    def __init__(self, max_posting_list: int):
        self.max_posting_list = max_posting_list
        self._postings: Dict[Any, List[str]] = {}
        self._banned: Set[Any] = set()

    def add(self, key: Any, entity_id: str) -> None:
        if key is None or key in self._banned:
            return
        bucket = self._postings.setdefault(key, [])
        bucket.append(entity_id)
        if len(bucket) > self.max_posting_list:
            self._banned.add(key)
            del self._postings[key]

    def get(self, key: Any) -> Sequence[str]:
        if key is None:
            return ()
        return self._postings.get(key, ())

    def posting_length(self, key: Any) -> int:
        return len(self._postings.get(key, ()))

    def __len__(self) -> int:
        return len(self._postings)


# ======================================================================================
# Block D -- sparse TF-IDF character n-gram retrieval (no dense matrix, no nested loops)
# ======================================================================================

class TfidfCharIndex:
    """Sparse char n-gram TF-IDF retrieval index for one text field of one source.

    Scalability:
      - Vocabulary is capped by ``max_features`` so memory is bounded regardless of
        corpus size (configurable; the vectorizer itself performs the document-frequency
        pruning, not a Python loop).
      - The corpus matrix is a scipy.sparse CSR matrix (n_docs x max_features), never a
        dense array.
      - Queries are matched with a single sparse-sparse matrix multiply per batch
        (query_batch @ corpus.T), and the resulting sparse row is reduced to its
        top_k via argpartition -- no S1 x S2/S3 Python nested loop, and no dense
        intermediate of shape (batch, n_docs) is ever materialized as a dense array.
      - Batching (``batch_size``) keeps the sparse similarity result's peak size
        bounded to one small batch of queries at a time.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 4),
        max_features: int = 50_000,
        min_df: int = 1,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self._vectorizer = None
        self._matrix = None
        self._ids: List[str] = []

    @property
    def is_built(self) -> bool:
        return self._matrix is not None and self._matrix.shape[0] > 0

    def build(self, texts: Sequence[str], ids: Sequence[str]) -> "TfidfCharIndex":
        if not texts:
            return self
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
        import numpy as np  # lazy import

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            dtype=np.float32,
        )
        self._matrix = self._vectorizer.fit_transform(texts)  # sparse CSR, n_docs x vocab
        self._ids = list(ids)
        return self

    def query_batches(
        self,
        query_texts: Sequence[str],
        top_k: int,
        min_similarity: float,
        batch_size: int,
    ) -> Iterator[List[Tuple[str, float]]]:
        """Yield, per query (in order), a list of (entity_id, similarity) pairs."""
        if not self.is_built or not query_texts:
            for _ in query_texts:
                yield []
            return

        import numpy as np  # lazy import

        for start in range(0, len(query_texts), batch_size):
            batch = query_texts[start:start + batch_size]
            query_matrix = self._vectorizer.transform(batch)  # sparse, batch x vocab
            similarity = (query_matrix @ self._matrix.T).tocsr()  # sparse, batch x n_docs
            for row_index in range(similarity.shape[0]):
                row = similarity.getrow(row_index)
                if row.nnz == 0:
                    yield []
                    continue
                data, indices = row.data, row.indices
                if row.nnz > top_k:
                    top_positions = np.argpartition(-data, top_k - 1)[:top_k]
                else:
                    top_positions = np.arange(row.nnz)
                pairs = [
                    (self._ids[indices[pos]], float(data[pos]))
                    for pos in top_positions
                    if data[pos] >= min_similarity
                ]
                pairs.sort(key=lambda pair: -pair[1])
                yield pairs


# ======================================================================================
# Per-source blocking index (Source 2 and Source 3 are always built/queried separately)
# ======================================================================================

class BlockingIndex:
    """All blocking-pass indexes for one source (S2 or S3), built with one streaming pass.

    Only the fields actually needed for blocking are retained per record
    (id, normalized_name, normalized_address text for TF-IDF corpora); no
    unnecessary raw columns are copied into the index, per requirement #13.
    """

    def __init__(self, source_label: str, config: BlockingConfig):
        self.source_label = source_label
        self.config = config

        self.exact_name = _PostingIndex(max_posting_list=1_000_000_000)  # Block A: no cap, deterministic
        self.token_index = _PostingIndex(max_posting_list=config.token_max_posting_list)  # Block B
        self.address_indexes: Dict[str, _PostingIndex] = {
            name: _PostingIndex(max_posting_list=config.address_key_max_posting_list)
            for name in (
                "country_postal", "country_city", "country_address_token",
                "postal_only", "city_address_token", "country_house_address_token",
            )
        }  # Block C
        self.country_token_indexes: Dict[str, _PostingIndex] = {}  # Block E, one _PostingIndex per country

        self._name_texts: List[str] = []
        self._name_ids: List[str] = []
        self._address_texts: List[str] = []
        self._address_ids: List[str] = []

        self.tfidf_name = TfidfCharIndex(
            ngram_range=config.tfidf_ngram_range,
            max_features=config.tfidf_max_features,
            min_df=config.tfidf_min_df,
        )
        self.tfidf_address = TfidfCharIndex(
            ngram_range=config.tfidf_ngram_range,
            max_features=config.tfidf_max_features,
            min_df=config.tfidf_min_df,
        )

        self.record_count = 0

    def __len__(self) -> int:
        return self.record_count

    # -- building -------------------------------------------------------------------

    def _significant_address_token(self, record: Mapping[str, Any]) -> Optional[str]:
        """A single representative address token, preferring street/locality hints
        the Normalizer already extracted (never re-derived from raw text)."""
        street = _first(_component(record, "street_indicators"))
        if street:
            return _key_norm(street)
        locality = _first(_component(record, "locality_candidates"))
        if locality:
            return _key_norm(locality)
        return None

    def build(self, records: Iterable[Mapping[str, Any]]) -> "BlockingIndex":
        cfg = self.config
        for record in records:
            entity_id = _record_id(record, cfg.id_field)
            self.record_count += 1

            normalized_name = record.get("normalized_name") or ""
            normalized_address = record.get("normalized_address") or ""

            # Block A -- exact normalized name
            if "exact_name" in cfg.enabled_blocks and normalized_name:
                self.exact_name.add(normalized_name, entity_id)

            # Block B -- name token blocking
            if "token" in cfg.enabled_blocks:
                for token in _name_tokens(record, cfg.token_min_length):
                    self.token_index.add(token, entity_id)

            # Block C -- address / component blocking (each key skipped gracefully if
            # its inputs are missing -- never assumes every record has every component)
            if "address" in cfg.enabled_blocks:
                country = _key_norm(_component(record, "country"))
                city = _key_norm(_component(record, "city"))
                postal = _key_norm(_first(_component(record, "postal_code")))
                postal_reliable = _confidence(record, "postal_code") != "low"
                house_no = _key_norm(_first(_component(record, "house_no_candidates")))
                addr_token = self._significant_address_token(record)

                if country and postal:
                    self.address_indexes["country_postal"].add((country, postal), entity_id)
                if country and city:
                    self.address_indexes["country_city"].add((country, city), entity_id)
                if country and addr_token:
                    self.address_indexes["country_address_token"].add((country, addr_token), entity_id)
                if postal and postal_reliable:
                    self.address_indexes["postal_only"].add(postal, entity_id)
                if city and addr_token:
                    self.address_indexes["city_address_token"].add((city, addr_token), entity_id)
                if country and house_no and addr_token:
                    self.address_indexes["country_house_address_token"].add(
                        (country, house_no, addr_token), entity_id
                    )

            # Block E -- country-partitioned token index (additive-only restriction pass)
            if "country_token" in cfg.enabled_blocks:
                country = _key_norm(_component(record, "country"))
                if country:
                    partition = self.country_token_indexes.setdefault(
                        country, _PostingIndex(max_posting_list=cfg.country_token_max_posting_list)
                    )
                    for token in _name_tokens(record, cfg.token_min_length):
                        partition.add(token, entity_id)

            # Block D corpora -- collected for a single fit at the end (see build_tfidf())
            if "tfidf_name" in cfg.enabled_blocks and normalized_name:
                self._name_texts.append(normalized_name)
                self._name_ids.append(entity_id)
            if "tfidf_address" in cfg.enabled_blocks and normalized_address:
                self._address_texts.append(normalized_address)
                self._address_ids.append(entity_id)

        return self

    def build_tfidf(self) -> "BlockingIndex":
        """Fit the TF-IDF retrieval indexes. Separate from build() so callers can
        persist/reuse the cheap indexes (A/B/C/E) without paying the TF-IDF fit cost
        when they don't need Block D, and so the corpora lists can be freed after."""
        if "tfidf_name" in self.config.enabled_blocks and self._name_texts:
            self.tfidf_name.build(self._name_texts, self._name_ids)
        if "tfidf_address" in self.config.enabled_blocks and self._address_texts:
            self.tfidf_address.build(self._address_texts, self._address_ids)
        # free the raw corpora; only the sparse matrices + id lists are needed afterward
        self._name_texts, self._name_ids = [], []
        self._address_texts, self._address_ids = [], []
        return self


# ======================================================================================
# Candidate generation -- unions the five blocking passes per S1 record
# ======================================================================================

BLOCK_NAMES = ("exact_name", "token", "address", "tfidf_name", "tfidf_address", "country_token")


class CandidateGenerator:
    """Runs all enabled blocking passes for S1 -> S2 and S1 -> S3, and unions them.

    S2 and S3 are always queried against their own, separately-built indexes
    (never mixed) and their per-block candidate sets are unioned per S1 record,
    per requirement #9.
    """

    def __init__(self, s2_index: BlockingIndex, s3_index: BlockingIndex, config: BlockingConfig):
        self.s2_index = s2_index
        self.s3_index = s3_index
        self.config = config

    # -- single-record, non-TF-IDF passes (O(1) dict lookups only) ------------------

    def _exact_name_for(self, index: BlockingIndex, record: Mapping[str, Any]) -> Set[str]:
        normalized_name = record.get("normalized_name") or ""
        if not normalized_name:
            return set()
        return set(index.exact_name.get(normalized_name))

    def _token_for(self, index: BlockingIndex, record: Mapping[str, Any]) -> Set[str]:
        cfg = self.config
        tokens = _name_tokens(record, cfg.token_min_length)
        if not tokens:
            return set()
        if cfg.token_use_rare_only:
            tokens = sorted(tokens, key=lambda tok: index.token_index.posting_length(tok))
        tokens = tokens[: cfg.token_max_per_record]
        candidates: Set[str] = set()
        for token in tokens:
            candidates.update(index.token_index.get(token))
        return candidates

    def _address_for(self, index: BlockingIndex, record: Mapping[str, Any]) -> Set[str]:
        country = _key_norm(_component(record, "country"))
        city = _key_norm(_component(record, "city"))
        postal = _key_norm(_first(_component(record, "postal_code")))
        postal_reliable = _confidence(record, "postal_code") != "low"
        house_no = _key_norm(_first(_component(record, "house_no_candidates")))
        addr_token = index._significant_address_token(record)

        candidates: Set[str] = set()
        if country and postal:
            candidates.update(index.address_indexes["country_postal"].get((country, postal)))
        if country and city:
            candidates.update(index.address_indexes["country_city"].get((country, city)))
        if country and addr_token:
            candidates.update(index.address_indexes["country_address_token"].get((country, addr_token)))
        if postal and postal_reliable:
            candidates.update(index.address_indexes["postal_only"].get(postal))
        if city and addr_token:
            candidates.update(index.address_indexes["city_address_token"].get((city, addr_token)))
        if country and house_no and addr_token:
            candidates.update(
                index.address_indexes["country_house_address_token"].get((country, house_no, addr_token))
            )
        return candidates

    def _country_token_for(self, index: BlockingIndex, record: Mapping[str, Any]) -> Set[str]:
        cfg = self.config
        country = _key_norm(_component(record, "country"))
        if not country:
            return set()  # missing/unseen country: contributes nothing, never eliminates
        partition = index.country_token_indexes.get(country)
        if partition is None:
            return set()
        tokens = _name_tokens(record, cfg.token_min_length)
        if cfg.token_use_rare_only:
            tokens = sorted(tokens, key=lambda tok: partition.posting_length(tok))
        tokens = tokens[: cfg.country_token_max_per_record]
        candidates: Set[str] = set()
        for token in tokens:
            candidates.update(partition.get(token))
        return candidates

    def _non_tfidf_blocks_for_record(
        self, index: BlockingIndex, record: Mapping[str, Any]
    ) -> Dict[str, Set[str]]:
        cfg = self.config
        blocks: Dict[str, Set[str]] = {}
        if "exact_name" in cfg.enabled_blocks:
            blocks["exact_name"] = self._exact_name_for(index, record)
        if "token" in cfg.enabled_blocks:
            blocks["token"] = self._token_for(index, record)
        if "address" in cfg.enabled_blocks:
            blocks["address"] = self._address_for(index, record)
        if "country_token" in cfg.enabled_blocks:
            blocks["country_token"] = self._country_token_for(index, record)
        return blocks

    # -- batched TF-IDF pass ---------------------------------------------------------

    def _tfidf_blocks_for_batch(
        self, index: BlockingIndex, records: Sequence[Mapping[str, Any]]
    ) -> Tuple[List[Set[str]], List[Set[str]]]:
        cfg = self.config
        n = len(records)
        name_results: List[Set[str]] = [set() for _ in range(n)]
        address_results: List[Set[str]] = [set() for _ in range(n)]

        if "tfidf_name" in cfg.enabled_blocks and index.tfidf_name.is_built:
            texts = [record.get("normalized_name") or "" for record in records]
            for i, pairs in enumerate(
                index.tfidf_name.query_batches(
                    texts, cfg.tfidf_name_top_k, cfg.tfidf_min_similarity, cfg.tfidf_batch_size
                )
            ):
                name_results[i] = {entity_id for entity_id, _ in pairs}

        if "tfidf_address" in cfg.enabled_blocks and index.tfidf_address.is_built:
            texts = [record.get("normalized_address") or "" for record in records]
            for i, pairs in enumerate(
                index.tfidf_address.query_batches(
                    texts, cfg.tfidf_address_top_k, cfg.tfidf_min_similarity, cfg.tfidf_batch_size
                )
            ):
                address_results[i] = {entity_id for entity_id, _ in pairs}

        return name_results, address_results

    # -- public API --------------------------------------------------------------

    def generate_batch(
        self, records: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Dict[str, Set[str]]]]:
        """For each S1 record, return {"S2": {block_name: ids}, "S3": {block_name: ids}}."""
        per_record: List[Dict[str, Dict[str, Set[str]]]] = [
            {"S2": self._non_tfidf_blocks_for_record(self.s2_index, record),
             "S3": self._non_tfidf_blocks_for_record(self.s3_index, record)}
            for record in records
        ]

        s2_tfidf_name, s2_tfidf_address = self._tfidf_blocks_for_batch(self.s2_index, records)
        s3_tfidf_name, s3_tfidf_address = self._tfidf_blocks_for_batch(self.s3_index, records)
        cfg = self.config
        for i in range(len(records)):
            if "tfidf_name" in cfg.enabled_blocks:
                per_record[i]["S2"]["tfidf_name"] = s2_tfidf_name[i]
                per_record[i]["S3"]["tfidf_name"] = s3_tfidf_name[i]
            if "tfidf_address" in cfg.enabled_blocks:
                per_record[i]["S2"]["tfidf_address"] = s2_tfidf_address[i]
                per_record[i]["S3"]["tfidf_address"] = s3_tfidf_address[i]
        return per_record

    @staticmethod
    def union_final(per_source_blocks: Dict[str, Dict[str, Set[str]]]) -> Set[str]:
        final: Set[str] = set()
        for source_blocks in per_source_blocks.values():
            for candidates in source_blocks.values():
                final |= candidates
        return final


# ======================================================================================
# candidate_pairs.tsv writer (matches utils/validate_submission.py's expected format)
# ======================================================================================

def write_candidate_pairs(
    path: str | Path,
    rows: Iterable[Tuple[str, Iterable[str]]],
) -> None:
    """Write candidate_pairs.tsv: header + one row per S1 id, comma-joined, deduped,
    deterministically sorted candidate ids -- exactly what validate_submission.py checks."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("source1_entity_id\tcandidate_entity_ids\n")
        for source1_id, candidate_ids in rows:
            ordered = sorted(set(candidate_ids))
            stream.write(f"{source1_id}\t{','.join(ordered)}\n")


def generate_candidate_pairs(
    s1_records: Iterable[Mapping[str, Any]],
    s2_index: BlockingIndex,
    s3_index: BlockingIndex,
    config: BlockingConfig,
    output_path: str | Path,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Stream S1 through the generator and write candidate_pairs.tsv directly
    (no intermediate blocking output is ever written as candidate_pairs.tsv --
    this is always the final, post-union, post-dedup set -- requirement #10)."""
    generator = CandidateGenerator(s2_index, s3_index, config)
    total_s1 = 0
    total_candidates = 0
    zero_candidate_s1 = 0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("source1_entity_id\tcandidate_entity_ids\n")
        batch: List[Mapping[str, Any]] = []

        def flush(batch_records: List[Mapping[str, Any]]) -> None:
            nonlocal total_s1, total_candidates, zero_candidate_s1
            if not batch_records:
                return
            per_record_blocks = generator.generate_batch(batch_records)
            for record, blocks in zip(batch_records, per_record_blocks):
                s1_id = _record_id(record, config.id_field)
                final = CandidateGenerator.union_final(blocks)
                total_s1 += 1
                total_candidates += len(final)
                if not final:
                    zero_candidate_s1 += 1
                stream.write(f"{s1_id}\t{','.join(sorted(final))}\n")

        for record in s1_records:
            batch.append(record)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        flush(batch)

    return {
        "s1_entities": total_s1,
        "s1_zero_candidates": zero_candidate_s1,
        "avg_candidates_per_s1": (total_candidates / total_s1) if total_s1 else 0.0,
    }


# ======================================================================================
# Training-data blocking evaluation
# ======================================================================================

def read_ground_truth(path: str | Path) -> Dict[str, Set[str]]:
    """Read a matching_results.tsv-shaped ground-truth file:
    source1_entity_id \t matched_entity_ids (comma-separated)."""
    ground_truth: Dict[str, Set[str]] = {}
    for row in read_tsv_records(path):
        s1_id = row.get("source1_entity_id")
        if not s1_id:
            continue
        raw = row.get("matched_entity_ids") or row.get("candidate_entity_ids") or ""
        ground_truth[s1_id] = {mid for mid in raw.split(",") if mid}
    return ground_truth


def _percentile(sorted_values: Sequence[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    lower, upper = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = k - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def evaluate_blocking(
    s1_records: Sequence[Mapping[str, Any]],
    s2_index: BlockingIndex,
    s3_index: BlockingIndex,
    config: BlockingConfig,
    ground_truth: Dict[str, Set[str]],
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Measure candidate recall, volume, and per-block contribution against ground truth."""
    generator = CandidateGenerator(s2_index, s3_index, config)

    per_block_recovered = {block: 0 for block in BLOCK_NAMES}
    union_recovered = 0
    total_true_matches = 0
    missed_matches: List[Tuple[str, str]] = []

    candidate_counts: List[int] = []
    zero_candidate_s1 = 0

    for start in range(0, len(s1_records), batch_size):
        batch = s1_records[start:start + batch_size]
        per_record_blocks = generator.generate_batch(batch)
        for record, blocks in zip(batch, per_record_blocks):
            s1_id = _record_id(record, config.id_field)
            truth = ground_truth.get(s1_id, set())

            union_candidates = CandidateGenerator.union_final(blocks)
            candidate_counts.append(len(union_candidates))
            if not union_candidates:
                zero_candidate_s1 += 1

            if not truth:
                continue
            total_true_matches += len(truth)

            per_block_union: Dict[str, Set[str]] = {block: set() for block in BLOCK_NAMES}
            for source_blocks in blocks.values():
                for block, ids in source_blocks.items():
                    per_block_union[block] |= ids

            union_recovered += len(truth & union_candidates)
            for match_id in truth - union_candidates:
                missed_matches.append((s1_id, match_id))
            for block in BLOCK_NAMES:
                per_block_recovered[block] += len(truth & per_block_union[block])

    candidate_counts_sorted = sorted(candidate_counts)
    n = len(candidate_counts_sorted)

    report: Dict[str, Any] = {
        "s1_entities": len(s1_records),
        "ground_truth_matches": total_true_matches,
        "recovered_true_matches": union_recovered,
        "candidate_recall": (union_recovered / total_true_matches) if total_true_matches else None,
        "average_candidates_per_s1": (sum(candidate_counts) / n) if n else 0.0,
        "median_candidates_per_s1": statistics.median(candidate_counts_sorted) if n else 0.0,
        "p95_candidates_per_s1": _percentile(candidate_counts_sorted, 0.95),
        "max_candidates_for_an_s1": candidate_counts_sorted[-1] if n else 0,
        "s1_entities_with_zero_candidates": zero_candidate_s1,
        "reduction_ratio": (
            1 - (sum(candidate_counts) / n) / max(len(s2_index) + len(s3_index), 1)
        ) if n else None,
        "missed_matches_sample": missed_matches[:50],
        "missed_matches_count": len(missed_matches),
        "per_block_recall": {
            block: (per_block_recovered[block] / total_true_matches) if total_true_matches else None
            for block in BLOCK_NAMES
        },
    }
    return report


# ======================================================================================
# Normalized-TSV I/O (reuses read_tsv_records/write_tsv_records; decodes the JSON columns
# that write_tsv_records serialized -- see normalizer.py's write_tsv_records)
# ======================================================================================

_JSON_COLUMNS = ("components", "numeric_features", "token_weights", "component_confidence", "tokens")


def read_normalized_tsv(path: str | Path, member: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    for row in read_tsv_records(path, member):
        for key in _JSON_COLUMNS:
            value = row.get(key)
            if isinstance(value, str) and value:
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        yield row


def iter_normalized_from_raw(
    raw_path: str | Path, artifact_path: str | Path, member: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """Go straight from a raw source TSV/ZIP through the existing (frozen) Normalizer's
    own streaming transform -- no intermediate normalized TSV is written."""
    from normalizer import Normalizer  # local import keeps this optional for pure-normalized-TSV usage

    normalizer = Normalizer.load(artifact_path)
    yield from normalizer.iter_transform(read_tsv_records(raw_path, member))


def _load_source(
    path: str, member: Optional[str], raw: bool, artifact: Optional[str]
) -> Iterator[Dict[str, Any]]:
    if raw:
        if not artifact:
            raise SystemExit("--raw requires --normalizer-artifact")
        return iter_normalized_from_raw(path, artifact, member)
    return read_normalized_tsv(path, member)


# ======================================================================================
# CLI
# ======================================================================================

def _build_config_from_args(args: argparse.Namespace) -> BlockingConfig:
    enabled = DEFAULT_ENABLED_BLOCKS
    if args.disable_block:
        enabled = frozenset(enabled - set(args.disable_block))
    return BlockingConfig(
        id_field=args.id_field,
        enabled_blocks=enabled,
        token_min_length=args.token_min_length,
        token_max_posting_list=args.token_max_posting_list,
        token_max_per_record=args.token_max_per_record,
        address_key_max_posting_list=args.address_key_max_posting_list,
        tfidf_name_top_k=args.tfidf_name_top_k,
        tfidf_address_top_k=args.tfidf_address_top_k,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_batch_size=args.tfidf_batch_size,
        tfidf_min_similarity=args.tfidf_min_similarity,
    )


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--s1", required=True, help="Normalized S1 TSV (or raw with --raw)")
    parser.add_argument("--s2", required=True, help="Normalized S2 TSV (or raw with --raw)")
    parser.add_argument("--s3", required=True, help="Normalized S3 TSV (or raw with --raw)")
    parser.add_argument("--s1-member", default=None, help="ZIP member for --s1 if it's a ZIP")
    parser.add_argument("--s2-member", default=None, help="ZIP member for --s2 if it's a ZIP")
    parser.add_argument("--s3-member", default=None, help="ZIP member for --s3 if it's a ZIP")
    parser.add_argument("--raw", action="store_true",
                         help="s1/s2/s3 are RAW source files; run them through the existing "
                              "Normalizer artifact given by --normalizer-artifact first")
    parser.add_argument("--normalizer-artifact", default=None,
                         help="Path to a saved Normalizer artifact (required with --raw)")
    parser.add_argument("--id-field", default="entity_id")
    parser.add_argument("--disable-block", action="append", choices=list(DEFAULT_ENABLED_BLOCKS),
                         help="Repeatable. Disable one of the five blocking passes.")
    parser.add_argument("--token-min-length", type=int, default=3)
    parser.add_argument("--token-max-posting-list", type=int, default=2000)
    parser.add_argument("--token-max-per-record", type=int, default=6)
    parser.add_argument("--address-key-max-posting-list", type=int, default=5000)
    parser.add_argument("--tfidf-name-top-k", type=int, default=30)
    parser.add_argument("--tfidf-address-top-k", type=int, default=30)
    parser.add_argument("--tfidf-max-features", type=int, default=50_000)
    parser.add_argument("--tfidf-batch-size", type=int, default=512)
    parser.add_argument("--tfidf-min-similarity", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512, help="S1 records processed per TF-IDF batch")


def _cmd_build(args: argparse.Namespace) -> None:
    config = _build_config_from_args(args)
    print(f"Building S2 index ({args.s2})...", file=sys.stderr)
    s2_records = list(_load_source(args.s2, args.s2_member, args.raw, args.normalizer_artifact))
    s2_index = BlockingIndex("S2", config).build(s2_records)
    s2_index.build_tfidf()
    del s2_records

    print(f"Building S3 index ({args.s3})...", file=sys.stderr)
    s3_records = list(_load_source(args.s3, args.s3_member, args.raw, args.normalizer_artifact))
    s3_index = BlockingIndex("S3", config).build(s3_records)
    s3_index.build_tfidf()
    del s3_records

    print(f"Generating candidates for S1 ({args.s1}) -> {args.output} ...", file=sys.stderr)
    s1_records = _load_source(args.s1, args.s1_member, args.raw, args.normalizer_artifact)
    stats = generate_candidate_pairs(
        s1_records, s2_index, s3_index, config, args.output, batch_size=args.batch_size
    )
    print(json.dumps(stats, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    config = _build_config_from_args(args)
    print(f"Building S2 index ({args.s2})...", file=sys.stderr)
    s2_records = list(_load_source(args.s2, args.s2_member, args.raw, args.normalizer_artifact))
    s2_index = BlockingIndex("S2", config).build(s2_records)
    s2_index.build_tfidf()
    del s2_records

    print(f"Building S3 index ({args.s3})...", file=sys.stderr)
    s3_records = list(_load_source(args.s3, args.s3_member, args.raw, args.normalizer_artifact))
    s3_index = BlockingIndex("S3", config).build(s3_records)
    s3_index.build_tfidf()
    del s3_records

    print(f"Loading ground truth ({args.ground_truth})...", file=sys.stderr)
    ground_truth = read_ground_truth(args.ground_truth)

    print(f"Evaluating S1 ({args.s1})...", file=sys.stderr)
    s1_records = list(_load_source(args.s1, args.s1_member, args.raw, args.normalizer_artifact))
    report = evaluate_blocking(
        s1_records, s2_index, s3_index, config, ground_truth, batch_size=args.batch_size
    )

    text = _format_report(report)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _format_report(report: Dict[str, Any]) -> str:
    def pct(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}%"

    lines = [
        "Blocking Evaluation",
        "-------------------",
        f"S1 entities: {report['s1_entities']}",
        f"Ground-truth matches: {report['ground_truth_matches']}",
        f"Recovered true matches: {report['recovered_true_matches']}",
        f"Candidate recall: {pct(report['candidate_recall'])}",
        "",
        f"Average candidates per S1: {report['average_candidates_per_s1']:.2f}",
        f"Median candidates per S1: {report['median_candidates_per_s1']:.0f}",
        f"P95 candidates per S1: {report['p95_candidates_per_s1']:.0f}",
        f"Max candidates for an S1: {report['max_candidates_for_an_s1']}",
        "",
        f"S1 entities with zero candidates: {report['s1_entities_with_zero_candidates']}",
        f"Reduction ratio: {pct(report['reduction_ratio'])}",
        "",
        "Recall by block:",
    ]
    for block, recall in report["per_block_recall"].items():
        lines.append(f"  {block:<16} {pct(recall)}")
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Generate candidate_pairs.tsv")
    _add_shared_args(build_parser)
    build_parser.add_argument("--output", required=True, help="Path to write candidate_pairs.tsv")
    build_parser.set_defaults(func=_cmd_build)

    eval_parser = subparsers.add_parser("evaluate", help="Blocking recall/volume evaluation")
    _add_shared_args(eval_parser)
    eval_parser.add_argument("--ground-truth", required=True, help="TSV: source1_entity_id, matched_entity_ids")
    eval_parser.add_argument("--output", default=None, help="Optional path to also write the report as JSON")
    eval_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()
