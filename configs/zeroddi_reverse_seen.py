_base_ = './zeroddi_seen.py'

model = dict(
    matching_mode='reverse',
    matching_hidden_dim=256,
    matching_dropout=0.1,
    matching_use_null_evidence=True,
    matching_use_evidence_gate=False,
)

work_dir = './work_dirs/zeroddi_reverse_seen'
