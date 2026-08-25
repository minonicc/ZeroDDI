_base_ = './new_s1_reverse_kg.py'

# Transfer the S0-selected N1-Nodes architecture unchanged to data split S1.
model = dict(
    leftmodel=dict(
        use_pharmacophore_pairs=True,
        pharmacophore_pair_dim=300,
        pharmacophore_pair_hidden_dim=300,
        pharmacophore_type_dim=32,
        pharmacophore_max_pairs=None,
        pharmacophore_pooling='sum',
        pharmacophore_selection_mode='drug_nodes',
        deduplicate_drugs_in_batch=False,
        cache_drug_graphs_on_device=True,
        batch_pharmacophore_pair_mlp=True,
    ),
    matching_use_pharmacophore_evidence=True,
    matching_pharmacophore_evidence_dim=300,
    matching_pharmacophore_hidden_dim=256,
    matching_pharmacophore_use_drug_nodes=True,
    matching_pharmacophore_top_k=None,
    matching_pharmacophore_drug_top_k=None,
    matching_pharmacophore_use_null_evidence=True,
    matching_use_pharmacophore_gate=False,
)

detect_anomaly = False
cpu_threads = 8
work_dir = './work_dirs/new_s1_pharmacophore_n1_nodes'
