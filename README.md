# Business Entity Resolution

## Current Stage

The full pipeline is implemented: normalization -> blocking -> candidate-pair
feature generation -> training-data labeling -> matching model -> entity-level
F0.5 evaluation/threshold selection -> test prediction -> `matching_results.tsv`
-> `utils/validate_submission.py`.

Each stage is a separate, independently testable module with its own CLI; none
of them force the full dataset into memory at once (see "Scalability" below).

## Architecture

```text
RAW DATA (train_source1/2/3.tsv, test_source1/2/3.tsv)
    |
    v
normalizer.py            fit/transform, artifact save/load           [unchanged]
    |
    v
blocking.py               multi-pass candidate generation             [unchanged]
    |  candidate_pairs.tsv
    v
features.py                pairwise feature computation                [this stage]
    |  pair_features.tsv (test) / used in-process for train
    v
training_data.py           + train_ground_truth.tsv -> labeled pairs   [this stage]
    |  labeled_pairs.tsv, entity-level train/val split
    v
matching_model.py          HistGradientBoostingClassifier (default)    [this stage]
    |  matcher_model.pkl
    v
evaluation.py               entity-level macro F0.5, threshold sweep    [this stage]
    |  matcher_metadata.json (chosen threshold)
    v
train_matcher.py            CLI wiring the four stages above for TRAIN  [this stage]
prediction.py               CLI: score TEST candidates -> results      [this stage]
    |  matching_results.tsv
    v
utils/validate_submission.py                                          [unchanged]
    |
    v
pipeline.py                 thin CLI wrapper over train/predict/validate
```

## Data and Generated Files

The challenge TSVs, fitted vocabulary, and transformed output files are intentionally not tracked in Git. The data files are hundreds of megabytes each, and generated outputs are several gigabytes. Obtain the challenge dataset through the team's authorized challenge distribution, then place the extracted `dataset/` directory at:

```text
6ab10eb3b23ba_student_resource/student_resource/dataset/
```

The ignored `artifacts/` and `output/` directories are created locally by the commands below. The repository does include the challenge description and `utils/validate_submission.py`.

The team shares generated normalization data in the [private Google Drive folder](https://drive.google.com/drive/folders/1hcKsEjXtJD7GkyHQa3BaKc-2I0sYbRWx?usp=sharing). Access is limited by the Drive folder's sharing settings; sign in with an invited team account.

## Requirements

- Python 3.10 or newer
- `pip install -r requirements.txt` (scikit-learn + numpy -- needed for blocking's
  TF-IDF pass and for the matching model; normalization/blocking's other passes
  are pure standard library)
- Run commands from this directory
- The challenge dataset must be obtained separately (see above)

The original `dataset.zip` is still damaged. The completed run used TSVs from `6ab10eb3b23ba_student_resource/student_resource/dataset/` directly, so no ZIP extraction is required to reproduce the transform with those files.

## 1. Fit and Transform (normalization)

To create the fitted artifact from all three training sources, use the Python `partial_fit` API below; the CLI `fit` command accepts one input TSV at a time. The artifact is generated locally and is not committed.

Transform the three training sources with the fitted artifact:

```powershell
foreach ($i in 1..3) {
  python normalizer.py transform `
    --input "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source$i.tsv" `
    --artifact artifacts/vocab_v1.json `
    --output "output/train_source${i}_normalized.tsv"
  if ($LASTEXITCODE -ne 0) { throw "Training transform failed for source $i" }
}
```

Transform each test source with the same artifact. Example for Source 1:

```powershell
python normalizer.py transform --input 6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv `
  --artifact artifacts/vocab_v1.json `
  --output output/test_source1_normalized.tsv
```

Repeat for `test_source2.tsv` and `test_source3.tsv`. Output rows retain the source fields and add derived values; nested fields are serialized as JSON in the TSV. Transform output is streamed.

## 2. Blocking (candidate generation)

Build candidate pairs separately for the train sources (needed to build training
data) and the test sources (needed for the final submission):

```bash
python3 blocking.py build \
    --s1 output/train_source1_normalized.tsv \
    --s2 output/train_source2_normalized.tsv \
    --s3 output/train_source3_normalized.tsv \
    --output output/train_candidate_pairs.tsv

python3 blocking.py build \
    --s1 output/test_source1_normalized.tsv \
    --s2 output/test_source2_normalized.tsv \
    --s3 output/test_source3_normalized.tsv \
    --output output/candidate_pairs.tsv
```

`blocking.py evaluate` (against `train_ground_truth.tsv`) reports candidate recall
and volume per blocking pass -- see `blocking.py --help` for every tunable
(`--tfidf-name-top-k`, `--token-max-posting-list`, `--disable-block`, etc.). This
stage is unchanged; see its own module docstring for details.

## 3. Train the matcher

`train_matcher.py` (also reachable as `pipeline.py train`) runs the whole training
side in one command: labels the train candidate pairs against ground truth
(`training_data.py`, with negative subsampling so the labeled table stays a bounded
size regardless of blocking's raw candidate volume), splits by **S1 entity** (never
by individual pair, to avoid leakage), trains the matcher, scores the validation
split, and sweeps thresholds against the **actual entity-level macro F0.5 metric**
(`evaluation.py`) -- never pairwise accuracy/F1, never top-1.

```bash
python3 train_matcher.py \
    --s1 output/train_source1_normalized.tsv \
    --s2 output/train_source2_normalized.tsv \
    --s3 output/train_source3_normalized.tsv \
    --candidate-pairs output/train_candidate_pairs.tsv \
    --ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv \
    --output-dir artifacts/matcher_v1 \
    --val-fraction 0.2 \
    --seed 42 \
    --max-negatives-per-s1 50
```

Writes to `artifacts/matcher_v1/`:
- `labeled_pairs.tsv` -- every labeled (S1, candidate) training row
- `matcher_model.pkl` -- the trained model (`matching_model.MatcherModel`)
- `matcher_metadata.json` -- selected threshold + validation macro F0.5/precision/recall

`--max-negatives-per-s1` bounds the labeled table's size (all positives are always
kept; negatives are randomly subsampled per S1 entity, reproducibly by `--seed`) --
this is the primary lever for keeping training tractable at full dataset scale,
since blocking's raw candidate volume (observed ~1355/S1 on the 10k/20k/20k sample)
would otherwise make the labeled table far larger than needed to train on.

## 4. Predict on the test set

```bash
python3 prediction.py \
    --s1 output/test_source1_normalized.tsv \
    --s2 output/test_source2_normalized.tsv \
    --s3 output/test_source3_normalized.tsv \
    --candidate-pairs output/candidate_pairs.tsv \
    --model artifacts/matcher_v1/matcher_model.pkl \
    --metadata artifacts/matcher_v1/matcher_metadata.json \
    --output output/matching_results.tsv
```

Streams one S1 entity at a time (only that entity's blocking-bounded candidate set
is ever in memory); every required S1 entity gets exactly one output row, including
zero-candidate and zero-prediction entities (empty `matched_entity_ids`).

## 5. Validate the submission

```bash
python3 pipeline.py validate \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

Equivalent to calling `utils/validate_submission.py` directly (that file is
unmodified; `pipeline.py validate` just shells out to it).

## Scalability decisions

- **Blocking** (unchanged): capped posting lists per key so no single common
  token/key can blow up a candidate list; TF-IDF retrieval is a sparse
  matrix-multiply per query batch, never a dense N x M matrix or a nested Python
  loop over all S1 x S2/S3 pairs.
- **Feature generation**: `features.iter_pair_feature_groups` processes one S1
  entity's candidate set at a time; nothing beyond that is held in memory.
- **Record lookup**: `features.RecordLookup` is an in-memory `id -> record` dict
  built once per source. This is correct and is what was used for the tested
  10k/20k/20k sample; it is **not** what should run against the full 4.6-5 GB
  normalized S2/S3 files on a 16 GB machine. Swapping it for a disk-backed or
  sort-merge-join implementation (without touching any feature-computation code
  above it, since `RecordLookup` is the one seam that isolates this concern) is
  the next large-scale optimization pass, not yet implemented.
- **Training-data volume**: `--max-negatives-per-s1` bounds the labeled table's
  size independent of blocking's raw candidate volume; all positives are always
  kept so recall of true matches into the training set is unaffected.
- **Model training**: `HistGradientBoostingClassifier` fits on an in-memory dense
  matrix -- after subsampling this is a bounded, reasonably small matrix, not the
  full candidate set. Retraining out-of-core (e.g. incremental fitting) is not
  implemented; it was judged unnecessary once negatives are subsampled.
- **Prediction**: same per-entity streaming as feature generation; the test-side
  `matching_results.tsv` write is a single pass with no buffering of the full
  result set.

## Testing

```bash
# from this directory (code/business_entity_resolution/) -- source lives in src/
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m py_compile src/normalizer.py src/blocking.py src/features.py src/training_data.py src/matching_model.py src/evaluation.py src/train_matcher.py src/prediction.py src/pipeline.py
```

(All commands above and in the numbered stages assume `src/` is on `PYTHONPATH`
and that you `cd src/` or pass `PYTHONPATH=src` before invoking any of the CLIs
listed in sections 1-5, since the modules import each other by plain module name,
e.g. `from blocking import read_normalized_tsv`.)

`Normalizer.validate()` reports empty-component and unknown-token ratios, and postal-pattern coverage when a postal pattern was learned. Validation thresholds can be configured in the method call. `blocking.py evaluate` reports candidate recall/volume against ground truth. `evaluation.py`'s macro F0.5 functions have dedicated unit tests for every edge case named in the challenge metric (zero true matches, zero predictions, multiple matches across S2+S3, etc.).

## Known limitations

- The 10k/20k/20k sample run (avg. ~1355 candidates/S1) suggests blocking's
  parameters are on the loose side; retuning `blocking.py`'s thresholds
  (`--token-max-posting-list`, `--tfidf-*-top-k`, etc.) before the full run would
  reduce both training-data volume and prediction-time cost. This is a tuning
  pass on the existing, unmodified blocking module, not a rewrite.
- `RecordLookup` is in-memory only (see Scalability above) -- the full 4.6-5 GB
  S2/S3 files have not yet been run through the matching stages end-to-end; the
  smoke tests here use small synthetic and the tested 10k/20k/20k sample.
- Threshold sweeping uses a configurable but by-default coarse grid
  (`evaluation._default_threshold_grid`, steps of 0.01); this is a correctness-
  first choice and cheap to make finer once wall-clock time on real data is known.
