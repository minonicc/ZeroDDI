_base_ = './new_s0_reverse_kg.py'

model = dict(
    leftmodel=dict(
        use_pharmacophore_pairs=True,
        pharmacophore_pair_dim=300,
        pharmacophore_pair_hidden_dim=300,
        pharmacophore_type_dim=32,
        pharmacophore_max_pairs=128,
        pharmacophore_pooling='sum',
        # Keep separate drug1/drug2 GNN passes so BatchNorm statistics and
        # per-occurrence Dropout behavior match the original training path.
        deduplicate_drugs_in_batch=False,
        cache_drug_graphs_on_device=True,
        batch_pharmacophore_pair_mlp=True,
    ),
    matching_use_pharmacophore_evidence=True,
    matching_pharmacophore_evidence_dim=300,
    matching_pharmacophore_hidden_dim=256,
)

work_dir = './work_dirs/new_s0_reverse_kg_pharmacophore'

# Anomaly detection is a debugging aid that forces extra synchronization.
# Disabling it does not change the model forward or gradient definitions.
detect_anomaly = False
