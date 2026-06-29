_base_ = './zeroddi.py'

model = dict(
    matching_mode='reverse',
    matching_hidden_dim=256,
    matching_dropout=0.1,
    matching_use_null_evidence=True,
    matching_use_evidence_gate=False,
    matching_use_kg_evidence=False,
    matching_kg_evidence_dim=300,
    matching_kg_hidden_dim=256,
    semantic_aux_lambda=0.3,
)

work_dir = './work_dirs/zeroddi_reverse'
