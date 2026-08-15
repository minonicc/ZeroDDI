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
| Struct-S4 | limited-step engineering run passed; controlled debug pending | 100 steps/epoch; alpha: 0 -> 0.001021 -> 0.001266 -> 0.003576 |
| T2-64 / T3-128 / T2-256 | implemented, CPU smoke passed | Engineering templates; formal base waits for stage-two winner |
| D3-12 / D3-16 | old D3-12 engineering run superseded; controlled debug pending | Earlier run used provisional mean pooling and the pre-correction pair path; it proves historical execution only |

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

Under the user-specified S0 test structure-selection protocol, P0
(128 + sum + no gate) remains the selected pharmacophore base. Struct-S1 and
Struct-S3 were rebased from the provisional P3 template to P0, explicitly
disable the gate, and passed the controlled config-difference audit before
their formal launches. Struct-S4 has no pharmacophore
branch and therefore cannot change with the stage-one pharmacophore winner; its
guard was removed after config-difference audit and its 100-epoch run started
on physical GPU 0 in tmux session `zeroddi_struct_s4_100`. Stage-three templates carry the
analogous `stage2_validation_winner` guard until stage-two structure selection.

Two optional stage-one factorial controls were added at commit `b236b62`:
P6 (128 + mean + gate) differs from P1 only by the pharmacophore gate, and P7
(512 + sum + gate) differs from P2 only by the gate. Both passed the controlled
config audit. Their intended physical GPUs are 1 and 2, but their tmux creation
requests were rejected by the execution permission layer. The user subsequently
deprioritized both optional controls, so they remain unrun and are marked
superseded rather than pending.

The 5-epoch stage-one debug metrics are saved in `stage1_debug_5ep.csv`.
They only establish correct execution and decreasing loss; they are not used for
method selection.

The 50-epoch validation-only screen is saved in
`stage1_screen50_validation.csv`. P3 wins by Macro-F1 under the predefined
selection rule, but its ACC and Kappa are lower than P1; the 100-epoch formal
runs must therefore report all three primary metrics before drawing a final
conclusion.

All three screen-best checkpoint files deserialize successfully on CPU and all
stored tensors are finite. P1 and P2 have identical 97-key state structures
(6,232,879 tensor elements), as required for a pooling/max-pair control. P3 has
exactly four additional keys, all belonging to the two linear layers of
`ReverseMatcher.pharmacophore_gate`, for 6,430,000 tensor elements; there are no
missing P1 keys or unrelated P3 additions. A strict current-model load is
deferred until the next model-build/debug boundary to avoid competing with the
three active formal data pipelines.

## Stage-one formal runs

The three schemes admitted by the predefined screen rule started their current
100-epoch seed-42 formal validation runs in detached tmux sessions on
2026-08-13. Each run uses Adam, learning rate 0.0001, batch size 128, and
validation Macro-F1 checkpoint selection. Earlier direct-exec attempts are
preserved as engineering traces but were superseded because the command host
terminated all three child processes after approximately three hours.

| Experiment | Physical GPU | Config | Work directory | Command override |
|---|---:|---|---|---|
| P1 | 4 | `configs/new_s0_reverse_kg_pharmacophore_p1_128_mean.py` | `work_dirs/formal100_v5_p1` | `--max-epochs 100` |
| P2 | 6 | `configs/new_s0_reverse_kg_pharmacophore_p2_512_sum.py` | `work_dirs/formal100_v5_p2` | `--max-epochs 100` |
| P3 | 7 | `configs/new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py` | `work_dirs/formal100_v5_p3` | `--max-epochs 100` |

Physical GPU 5 was occupied by an unrelated process and was left untouched.
All three runs passed initialization and reached at least step 100 of epoch 1
with finite, decreasing training loss.

Exact launch commands (each process sees its bound physical GPU as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/wumengying/miniconda3/envs/zeroddi/bin/python main.py --config configs/new_s0_reverse_kg_pharmacophore_p1_128_mean.py --work-dir work_dirs/formal100_v5_p1 --device cuda:0 --seednumber 42 --max-epochs 100
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/wumengying/miniconda3/envs/zeroddi/bin/python main.py --config configs/new_s0_reverse_kg_pharmacophore_p2_512_sum.py --work-dir work_dirs/formal100_v5_p2 --device cuda:0 --seednumber 42 --max-epochs 100
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/wumengying/miniconda3/envs/zeroddi/bin/python main.py --config configs/new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py --work-dir work_dirs/formal100_v5_p3 --device cuda:0 --seednumber 42 --max-epochs 100
```

The v5 training-time validation path computes ACC, Kappa, Macro-F1,
Weighted-F1, Macro-Precision, and Macro-Recall on every epoch. PR-AUC is an
auxiliary metric and is deliberately deferred because three prior runs were
observed directly inside scikit-learn `average_precision_score` for tens of
minutes. Explicit checkpoint evaluation retains the default full metric path
and therefore computes both macro and micro PR-AUC once for the final report.
This does not alter checkpoint selection, which never uses PR-AUC.

P2 and P3 completed exact ordered validation epochs 1--100. P2 selected epoch
99 (validation ACC 0.927197, Kappa 0.920184, Macro-F1 0.835698); P3 selected
epoch 84 (validation ACC 0.917562, Kappa 0.909645, Macro-F1 0.848417). Their
validation-selected S0 test results are recorded separately as `finaltest`
rows: P2 ACC/Kappa/Macro-F1 = 0.927150/0.920189/0.834817 and P3 =
0.916101/0.908090/0.821463.

P1 subsequently completed exact epochs 1--100 and selected epoch 95
(validation ACC/Kappa/Macro-F1 = 0.923996/0.916744/0.829974). Its corresponding
test values are 0.925342/0.918265/0.832006, with macro/micro PR-AUC
0.896969/0.977814. The validation-only decision file
`stage1_formal100_winner.json` is retained as an audit of the automatic
validation-Macro-F1 decision only: it selects P3 and flags that P3 has lower ACC
and Kappa than both alternatives. It is not the structure-selection authority.
The authoritative decision is `stage1_test_structure_selection.json`, which
compares the final S0 test results under the protocol explicitly selected by the
user and retains the pre-existing P0 result as the pharmacophore base. None of
P1/P2/P3 improves overall test behavior over P0, and none establishes the
stable new overall best required to unlock S2/S3 dataset-split evaluation.

Struct-S4 also completed exact epochs 1--100 and selected epoch 100. Its
validation ACC/Kappa/Macro-F1 are 0.919065/0.911319/0.785700; the corresponding
test values are 0.919130/0.911441/0.794708. The learned residual coefficient
grew monotonically from 0 to 11.060376. Although training remained finite,
this is an abnormally large residual scale and is evidence that the unbounded
fixed-substructure residual dominates rather than remaining a small correction.
The observation is reported as a diagnostic conclusion; it is not used to
alter this completed experiment after viewing test results.

After P0 was locked by `stage1_test_structure_selection.json`, Struct-S1 and
Struct-S3 were rebased to 128 + sum + no gate, audited, and launched directly
as 100-epoch seed-42 runs per the user-specified stage-two protocol. Struct-S1
runs in tmux session `zeroddi_struct_s1_p0_100` on physical GPU 4 with work
directory `work_dirs/formal100_struct_s1_p0`; Struct-S3 runs in
`zeroddi_struct_s3_p0_100` on physical GPU 5 with work directory
`work_dirs/formal100_struct_s3_p0`. Both use commit `7232a00`, eight CPU/MKL
threads, and validation Macro-F1 checkpoint selection before S0 test structure
comparison.

Struct-S1 and Struct-S3 completed exact epochs 1--100. Struct-S1 selected
validation epoch 95 with ACC/Kappa/Macro-F1
0.914247/0.906093/0.803498; its S0 test values are
0.913894/0.905755/0.804786. Struct-S3 selected validation epoch 97 with
0.915178/0.907025/0.802926; its S0 test values are
0.914912/0.906815/0.818210. Struct-S3 is higher than Struct-S1 on all three
test metrics, but neither improves on the P0/B0 structure reference. Its
fixed-substructure residual coefficient reached -10.974743, mirroring the
large-magnitude residual observed for Struct-S4. Final test evaluations used
physical GPUs 0 and 1 and the validation-selected checkpoints; complete
PR-AUC outputs remain in their evaluation logs.

An additional intermediate pair-count control, P8 (256 + sum + no gate), was
added at commit `1b4c678`. Its audit proves that it differs from P2 only in
`pharmacophore_max_pairs` (256 versus 512). The 100-epoch seed-42 run uses
physical GPU 6, tmux session `zeroddi_p8_256_sum_100`, and work directory
`work_dirs/formal100_p8_256_sum`. A representative five-epoch T3-128 engineering
debug was also proposed for idle GPU 7, but its tmux creation permission was
rejected before launch; no stage-three result is claimed from that request.

The stage-three representative comparison was subsequently launched directly
for 100 epochs from the P0-controlled T3-128 architecture. Session
`zeroddi_t3_top128_softmax_100` uses physical GPU 7 and retains the specified
softmax over 128 candidate-specific real pairs plus the pharmacophore null
token. Session `zeroddi_t3_top128_sigmoid_100` uses physical GPU 3 and instead
assigns independent sigmoid relevance weights, divides their weighted sum by
the number of valid selected pairs (not the weight sum), and disables the null
token only for the pharmacophore branch. The latter therefore permits an
all-irrelevant selected set to approach zero evidence while leaving the
substructure and KG null-token behavior unchanged. Both runs use seed 42,
commit `3380396`, eight CPU/MKL threads, and work directories
`work_dirs/formal100_t3_top128_softmax` and
`work_dirs/formal100_t3_top128_sigmoid_mean`, respectively. Physical GPU 3
also had an unrelated approximately 3.5 GiB process; it was left untouched,
and the training job was co-located only because sufficient A100 memory
remained and the user explicitly assigned GPU 3.

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

Diagnostic scalar aggregation changed after the running stage-one jobs were
launched: P1/P2/P3 use the historical mean-of-batch-means values, whereas new
jobs use exact element weighting. The summarizer therefore requires one
`--semantics NAME legacy_batch_mean|element_weighted` declaration for every
`--run`; it writes `aggregation_semantics` into every output row and rejects
missing, extra, duplicate, or unknown declarations. These two scalar-statistic
semantics must not be interpreted as model effects. The underlying training
outputs, validation metrics, attention computation, and checkpoint selection
are unchanged.

The machine-readable unified table is `pharmacophore_results.csv`. It keeps
historical test references, validation screens, formal validation runs, and
eventual final tests in separate `metric_split` / `run_level` fields. Formal
summaries use `tools/summarize_validation_runs.py --expected-epochs 100`, which
requires the exact ordered validation epoch sequence 1--100. It refuses not
only a short run but also logs with missing, duplicated, unexpected, or
out-of-order validation epochs, preventing line-count coincidences from being
reported as a finished formal run. Its checkpoint comparator is covered at the
strict 0.001 Macro-F1 boundary and matches the training comparator.
`tools/audit_pharmacophore_results.py` additionally enforces the required
experiment/run-level rows, unique per-seed records, complete primary-metric
triples, validation-only preselection rows, test-only final rows, and explicit
checkpoint paths for running formal experiments. Superseded engineering runs
and their required controlled reruns therefore remain separate machine-readable
records.

After all formal logs pass the exact 100-epoch completeness check,
`tools/select_validation_winner.py` converts that validation summary into an
auditable JSON decision. It forms the strict Macro-F1 band using
`max(Macro-F1) - candidate < 0.001`, then resolves candidates in that band by
Kappa and ACC. It also flags a selected run whose Macro-F1 gain accompanies
lower Kappa and ACC for explicit trade-off review. The decision consumes no
test metrics and does not mutate or unlock dependent configs: the winner still
has to be reviewed, rebased into the Struct templates, audited, and only then
have the provisional dependency marker removed.

The running P1/P2/P3 processes exposed an evaluation accumulation bottleneck:
each validation batch repeatedly appended to every prior NumPy result and also
retained the model's large `instance` and `prototype` outputs even though no
evaluation branch consumes them. This produced quadratic copying and roughly
33 GB resident CPU memory per process near epoch-10 validation. The running
jobs were not interrupted and therefore retain their original code and metric
semantics. Subsequent processes collect prediction and label batches in lists,
concatenate each exactly once, and discard the two unused outputs. The
classification inputs and metric functions are unchanged. A smoke model whose
unused outputs deliberately fail on materialization confirms they are not
retained and that ACC, Kappa, and Macro-F1 remain exact on a multi-batch input.

Formal training logs are epoch-oriented again: batch progress is disabled by
default and can be enabled only with an explicit positive
`train_progress_interval` in a debug config. The inherited generic
`log_config.interval=50` still controls the framework logger configuration but
no longer emits roughly 67 `epoch:N step:M/3353` lines per training epoch.
This changes logging only, not optimization, validation, or checkpointing.

### Formal restart after abnormal validation

The first formal processes were user-authorized to stop with `SIGTERM` on
2026-08-12 after epoch-10 validation remained active for more than 80 minutes.
They had completed nine validation epochs, had not reached the epoch-20
checkpoint boundary, and produced no resumable optimizer/epoch checkpoint.
Their work directories and logs remain preserved as abnormal-run evidence:
`formal100_p1`, `formal100_p2`, and `formal100_p3`. They are excluded from all
formal summaries and winner decisions.

Fresh seed-42 runs use commit `849cafc`, which includes linear validation
accumulation (`0e42b6c`) and concise formal logs. They were deliberately
staggered and use new work directories:

| Experiment | Physical GPU | Start | Session | PID | Work directory |
|---|---:|---|---:|---:|---|
| P1 | 4 | 16:31 | 70834 | 51325 | `work_dirs/formal100_v2_p1` |
| P2 | 6 | 16:35 | 93788 | 70387 | `work_dirs/formal100_v2_p2` |
| P3 | 7 | 16:38 | 13780 | 82711 | `work_dirs/formal100_v2_p3` |

Each command otherwise retains the original config, seed 42, 100 epochs,
Adam learning rate 0.0001, batch size 128, and validation-only checkpoint rule.
P1 and P2 reached the 429,137-instance training loop before this record; P3
was still completing initialization. Physical GPU 5 was not assigned.

The v2 runs exposed severe host CPU oversubscription rather than a deterministic
metric regression: the same 61,234-by-197 classification metrics that took
16--20 seconds in `new_s0_reverse_kg_gnn/20260804_153833.log` varied from about
12 seconds to more than an hour while system load reached 177, around 280 tasks
were runnable, and system CPU time approached 47%. The repository had never
set PyTorch/MKL thread limits; this environment defaults to 255 intra-op and
127 inter-op threads, with about 321 native threads visible after full model
initialization. V2 P1/P2/P3 were therefore user-authorized to stop after four,
three, and four partial epochs respectively and remain excluded from selection.

Subsequent launches set both `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8`, while
the resolved S0 config records `cpu_threads=8`; `main.py` also enforces eight
PyTorch intra-op threads and one inter-op coordinator. A synthetic
61,234-by-197 full metric calculation, including macro and micro PR-AUC, took
12.0 seconds and stayed at eight native threads under these limits. The limits
change scheduling/resource use only, not logits or metric definitions.

Large P3 per-class gate arrays remain fully serialized in diagnostics JSONL,
but the text log now emits only scalar diagnostic summaries. This prevents the
197-by-10 gate histogram and 197-value class mean from flooding the main log.
