_base_ = './zeroddi.py'

model = dict(
    matching_mode='reverse',
    matching_hidden_dim=256,
    matching_dropout=0.1,
)

work_dir = './work_dirs/zeroddi_reverse'
