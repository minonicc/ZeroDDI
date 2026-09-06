# KG GraphSAGE + N1 pharmacophore experiment

Sources: `codex/goal-experiment` (6cb83e3) and `codex/kg-content-update` (2725e1f).
Branch: `codex/kg-gnn-pharmacophore`.

The final N1 architecture concatenates A/B typed pharmacophore nodes and uses
candidate DDIE attention with null evidence. Hard Top-K, pair products, and
pharmacophore gates are disabled. Historical variants remain available only
through their original configs.

KG uses the v2 SQLite pair graphs, entity/type/A-distance/B-distance embeddings,
and one edge-aware mean GraphSAGE layer with relation embeddings. Its output
is queried by each DDIE. Substructure, KG, and pharmacophore summaries are
concatenated for the candidate scorer.

Configs: `configs/new_s{0,1,2}_kg_gnn_pharmacophore_n1.py`.
100 epochs, seed 42, batch 128, CPU threads 8; validation Macro-F1 checkpoint
selection inherited from N1. Batch 128 matches the flat KG and final N1 configs.
Historical KG GNN configs use batch 64, so comparisons against those runs also
include a batch-size change. For a controlled ablation, rerun KG GNN at batch
128 with the same split, seed, and other training settings.

Run from this worktree: `bash tools/run_kg_pharm_pipeline.sh s0 1`.
The pipeline trains, then evaluates the validation-selected checkpoint on the
seen-label test set. S1/S2 use the same architecture and separate output paths.
Existing data are linked from the original repository; outputs are local.

Validation: reverse-attention smoke suite including joint KG+N1 forward/backward
and gradients through both encoders; pharmacophore encoder smoke; validation
selection smoke; resolved S0/S1/S2 configs and v2 graph metadata checked.
No combined accuracy result is available before training and evaluation finish.
