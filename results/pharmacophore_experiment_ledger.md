# Pharmacophore experiment ledger

## Protocol audit

- Training entry point: `main.py`, which calls `tools.train.train_model`.
- S0 base chain: `new_s0_reverse_kg*.py` -> `new_s0_reverse_kg.py` ->
  `zeroddi_reverse_kg_seen.py` -> `zeroddi_reverse_seen.py` -> `zeroddi_seen.py`.
- S0 files: `data/KnowDDI/drugbank_true_s0/{train,val,test}.csv` with
  429137 / 61234 / 122814 rows and 197 classes.
- Seed: 42; optimizer: Adam; learning rate: 0.0001; batch size: 128;
  standard full run: 100 epochs.
- Model selection uses the complete validation split and compares Macro-F1,
  then Kappa and ACC when Macro-F1 differs by less than 0.001.
- New validation improvements are persisted immediately to the configured best
  checkpoint path; periodic epoch snapshots and the final save remain as
  additional recovery points. Runs launched before this change retain the
  original every-20-epoch/final persistence schedule but use the same selection
  rule.
- Test is not invoked by the training path. It is reserved for a selected
  validation checkpoint via the explicit evaluation path.
- The explicit S0 seen-label test path intentionally builds `data.zsl_test`:
  dataset-local class IDs are assigned by first appearance, and train versus
  test differ at 195 of 197 class-order positions. `data.zsl_test` supplies
  test-local prototypes and KG tokens consistently; substituting `test_seen`
  (which uses train prototypes and has no KG file) would misalign labels. The
  `--seen_para` path writes per-class metrics and the confusion matrix.
- The inherited evaluator at commit `bb929c1` used `RandomSampler` together
  with `drop_last=True`, so every evaluation randomly omitted the final partial
  batch (50 of 61234 S0 validation rows). The current branch evaluates all rows
  deterministically. Historical P0/P4/P5 numbers therefore remain useful
  references but are not bit-for-bit comparable unless their checkpoints are
  re-evaluated with the corrected evaluator; those checkpoints are not present
  in this worktree.
- Repository split configs are literally named `new_s0_reverse_kg.py`,
  `new_s1_reverse_kg.py`, and `new_s2_reverse_kg.py`, backed by directories
  `drugbank_true_s0`, `drugbank_true_s1`, and `drugbank_true_s2`. There is no
  literal S3 config in the checked-out repository; the requested S2/S3 naming
  must be mapped from authoritative project provenance before cross-split runs.

## Historical test references supplied with the task

| ID | Seed | ACC | Kappa | Macro-F1 | Configuration |
|---|---:|---:|---:|---:|---|
| B0 | 42 | 0.927252 | 0.920359 | 0.840984 | Original best model |
| P0 | 42 | 0.925631 | 0.918534 | 0.843648 | 128, sum, no gate |
| P4 | 42 | 0.920547 | 0.913024 | 0.833668 | 512, mean, no gate |
| P5 | 42 | 0.920433 | 0.912914 | 0.843138 | 512, mean, gate |

These are historical references only and are not used to tune against the test
set.

## S0 coverage diagnostics

Full report: `pharmacophore_coverage_s0.json`.

- 2151 valid drugs; median 12 and mean 14.08 pharmacophores per drug.
- 613185 drug pairs; median 133, mean 177.65, p95 441, maximum 28424 pairs.
- Fraction above 64 / 128 / 144 / 256 / 512 pairs:
  83.11% / 52.61% / 44.58% / 16.81% / 3.52%.
- Mean node coverage of a fixed row-major prefix at 128 / 256 / 512 pairs:
  88.08% / 95.04% / 97.05%.
- Mean pharmacophore-type coverage at 128 / 256 / 512 pairs:
  95.13% / 96.91% / 97.41%.

## Experiment status

Structure experiment names are always written as Struct-S1, Struct-S3, and
Struct-S4 so they cannot be confused with dataset split names.

| Experiment | State | Notes |
|---|---|---|
| P1: 128 mean no gate | 50-epoch screen complete | Best epoch 49: ACC 0.901640, Kappa 0.892267, Macro-F1 0.740629 |
| P2: 512 sum no gate | 50-epoch screen complete | Best epoch 49: ACC 0.901362, Kappa 0.891847, Macro-F1 0.726708 |
| P3: 128 sum gate | 50-epoch screen winner | Best epoch 50: ACC 0.895026, Kappa 0.885096, Macro-F1 0.762781 |
| Struct-S1 | implementation smoke passed; P3 controls locked | Earlier 3-epoch limited-step GPU debug used the provisional P0 controls and is engineering evidence only; rerun required before ranking |
| Struct-S3 | implemented, CPU smoke passed | Formal config now inherits stage-one winner P3 |
| Struct-S4 | 3-epoch limited-step GPU debug passed | Alpha: 0 -> 0.001021 -> 0.001266 -> 0.003576 |
| T2-64 / T3-128 / T2-256 | implemented, CPU smoke passed | Engineering templates; formal base waits for stage-two winner |
| D3-12 / D3-16 | D3-12 optimized limited-step GPU debug passed | Earlier debug used provisional mean pooling and proves execution only; controlled templates now retain inherited pooling |

The 5-epoch stage-one debug metrics are saved in `stage1_debug_5ep.csv`.
They only establish correct execution and decreasing loss; they are not used for
method selection.

The 50-epoch validation-only screen is saved in
`stage1_screen50_validation.csv`. P3 wins by Macro-F1 under the predefined
selection rule, but its ACC and Kappa are lower than P1; the 100-epoch formal
runs must therefore report all three primary metrics before drawing a final
conclusion.

## Stage-one formal runs

The three schemes admitted by the predefined screen rule started 100-epoch
seed-42 formal validation runs on 2026-08-12. Each run uses Adam, learning rate
0.0001, batch size 128, and validation Macro-F1 checkpoint selection.

| Experiment | Physical GPU | Config | Work directory | Command override |
|---|---:|---|---|---|
| P1 | 4 | `configs/new_s0_reverse_kg_pharmacophore_p1_128_mean.py` | `work_dirs/formal100_p1` | `--max-epochs 100` |
| P2 | 6 | `configs/new_s0_reverse_kg_pharmacophore_p2_512_sum.py` | `work_dirs/formal100_p2` | `--max-epochs 100` |
| P3 | 7 | `configs/new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py` | `work_dirs/formal100_p3` | `--max-epochs 100` |

Physical GPU 5 was occupied by an unrelated process and was left untouched.
All three runs passed initialization and reached at least step 100 of epoch 1
with finite, decreasing training loss.

Epoch-level evidence diagnostics can be flattened with
`tools/summarize_pharmacophore_diagnostics.py`. The script combines diagnostics
JSONL with alpha values from training logs, preserves class/type arrays as JSON,
and adds mean, standard deviation, minimum, and maximum summaries for direct
run-to-run comparison. New runs also record a normalized 10-bin gate histogram
and the fractions at or below 0.05 and at or above 0.95 so gate saturation can
be detected directly. Final per-class CSV files include both the split-local
class index and the corresponding DDIE event ID.

The machine-readable unified table is `pharmacophore_results.csv`. It keeps
historical test references, validation screens, formal validation runs, and
eventual final tests in separate `metric_split` / `run_level` fields. Formal
summaries use `tools/summarize_validation_runs.py --expected-epochs 100`, which
refuses to summarize an incomplete run as a finished 100-epoch result.
