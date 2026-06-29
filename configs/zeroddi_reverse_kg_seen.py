_base_ = './zeroddi_reverse_seen.py'

model = dict(
    matching_use_kg_evidence=True,
    matching_kg_evidence_dim=300,
    matching_kg_hidden_dim=256,
)

kg_pair_file = './data/DrugBank5.1.9/kg/molecbionet_pair_kg_seen.json'

data = dict(
    train=dict(kg_pair_file=kg_pair_file, kg_max_tokens=128),
    zsl_val=dict(kg_pair_file=kg_pair_file, kg_max_tokens=128),
    gzsl_val=dict(kg_pair_file=kg_pair_file, kg_max_tokens=128),
    zsl_test=dict(kg_pair_file=kg_pair_file, kg_max_tokens=128),
    gzsl_test=dict(kg_pair_file=kg_pair_file, kg_max_tokens=128),
)

work_dir = './work_dirs/zeroddi_reverse_kg_seen'
