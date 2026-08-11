_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(
        pharmacophore_max_pairs=128,
        pharmacophore_pooling='mean',
    ),
    matching_use_pharmacophore_gate=False,
)

work_dir = './work_dirs/new_s0_reverse_kg_pharmacophore_p1_128_mean'
