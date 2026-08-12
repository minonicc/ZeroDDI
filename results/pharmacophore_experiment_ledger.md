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
  checkpoint path together with a `.metrics.json` sidecar containing the best
  epoch, seed, validation metrics, and selection rule; periodic epoch snapshots
  and the final save remain as additional recovery points. Runs launched before
  this change retain the
  original every-20-epoch/final persistence schedule but use the same selection
  rule and have no sidecar; their best epoch must be audited from the complete
  validation log.
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

All three Struct configs set `semantic_aux_lambda=0.0` as a necessary
consequence of removing the DDIE-query substructure branch. In the inherited
classifier, this auxiliary term calls `Local(left_output, sub_structure, ...)`;
`Local` uses `sub_structure` as the query in candidate-DDIE semantic attention.
It is therefore part of the removed query-substructure mechanism rather than
an independent regularizer that could be retained after `sub_structure=None`.
The Struct comparisons all apply this removal consistently.

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

The current D3 implementation follows the specified operation order exactly:
it pools each drug's pharmacophore nodes, performs candidate-specific node
selection, and constructs/encodes only the selected K x K pairs. It no longer
precomputes the complete Cartesian pair tensor merely to gather the selected
rows. The original available-pair count is passed separately as a small tensor,
so coverage and selection-fraction diagnostics retain their full-set
denominator without the full-set memory cost. The earlier D3-12 engineering
debug predates this correction and must therefore be rerun before screening.
For a controlled comparison, the direct D3 path reuses the Leftmodel
pharmacophore type embedding, pair MLP, and LayerNorm rather than initializing a
second pair encoder inside the selector. A deterministic backward smoke check
proves that D3 loss gradients reach that registered shared encoder, while the
selector owns only the new DDIE-guided node-selection and pair-aggregation
parameters. A save/load smoke check also verifies that the shared pair encoder
appears only once in the parent state dict, strict loading has no missing or
unexpected keys, and the restored selector references its restored Leftmodel
encoder.

Pair-level Top-K must rank the complete real pair set, so the T2/T3 templates
intentionally set `pharmacophore_max_pairs=None`; silently capping them would
change the method. A resource audit of the S0 training rows found 30 / 429137
pairs above 5000 pharmacophore pairs, 3 above 10000, and 1 at the observed
maximum of 28424. The current encoder pads a batch to its largest pair set, so
the controlled 3--5 epoch GPU debug must explicitly cover peak memory. If that
debug encounters a memory failure, the remedy must be output-equivalent packed
or chunked processing, not a smaller candidate cap or changed batch size in the
screen comparison.

The Struct-S4 alpha evidence is end-to-end rather than inferred from source
alone. Its effective config records `fixed_substructure_alpha_init=0.0`; the
three GPU debug epochs log 0.001021, 0.001266, and 0.003576. The
validation-selected checkpoint contains the registered state-dict key
`Leftmodel.fixed_substructure_alpha` with value 0.0010207274463027716, matching
the best epoch's log. Since the training optimizer is constructed from all
`model.named_parameters()`, the nonzero trajectory proves the parameter was in
the optimizer, received a gradient through the residual path, was updated, and
survived checkpoint serialization.

The fixed-substructure path is also DDIE-independent by construction. In
`SubExtractor`, the 30 substructure queries are learned model parameters and
attend only to atom-GNN keys/values; candidate DDIE representations are not an
input. Each resulting substructure token is normalized, then the token axis is
mean-pooled before the shared scalar alpha residual is added separately to each
drug's 300-dimensional global representation. The global and substructure
representations are both 300-dimensional in the current GNN, so the configured
identity projection is dimensionally exact; a learned projection is not needed
for this architecture. Struct-S1 sets `use_sub=False`, so neither this fixed
path nor the old queried-substructure tokens are computed.

P3 remains only the provisional 50-epoch screen winner for the checked-in
Struct-S1/S3 templates. Those templates may be used for controlled engineering
debugs, but must be rebased if the stage-one 100-epoch validation winner changes
before any Struct screen or formal run is launched.

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

Exact launch commands (each process sees its bound physical GPU as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=4 /home/wumengying/miniconda3/envs/zeroddi/bin/python main.py --config configs/new_s0_reverse_kg_pharmacophore_p1_128_mean.py --work-dir work_dirs/formal100_p1 --device cuda:0 --seednumber 42 --max-epochs 100
CUDA_VISIBLE_DEVICES=6 /home/wumengying/miniconda3/envs/zeroddi/bin/python main.py --config configs/new_s0_reverse_kg_pharmacophore_p2_512_sum.py --work-dir work_dirs/formal100_p2 --device cuda:0 --seednumber 42 --max-epochs 100
CUDA_VISIBLE_DEVICES=7 /home/wumengying/miniconda3/envs/zeroddi/bin/python main.py --config configs/new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py --work-dir work_dirs/formal100_p3 --device cuda:0 --seednumber 42 --max-epochs 100
```

Epoch-level evidence diagnostics can be flattened with
`tools/summarize_pharmacophore_diagnostics.py`. The script combines diagnostics
JSONL with alpha values from training logs, preserves class/type arrays as JSON,
and adds mean, standard deviation, minimum, and maximum summaries for direct
run-to-run comparison. New runs also record a normalized 10-bin gate histogram
and the fractions at or below 0.05 and at or above 0.95 so gate saturation can
be detected directly. All scalar diagnostics use exact element-count weighting,
so a short final batch is not given the same weight as a full batch; gate
population standard deviation is likewise computed from global sums and squared
sums. Final per-class CSV files include both the split-local class index and the
corresponding DDIE event ID.

The machine-readable unified table is `pharmacophore_results.csv`. It keeps
historical test references, validation screens, formal validation runs, and
eventual final tests in separate `metric_split` / `run_level` fields. Formal
summaries use `tools/summarize_validation_runs.py --expected-epochs 100`, which
requires the exact ordered validation epoch sequence 1--100. It refuses not
only a short run but also logs with missing, duplicated, unexpected, or
out-of-order validation epochs, preventing line-count coincidences from being
reported as a finished formal run. Its checkpoint comparator is covered at the
strict 0.001 Macro-F1 boundary and matches the training comparator.
