_base_ = './new_s0_reverse_kg_pharmacophore_p1_128_mean.py'

model = dict(
    matching_use_pharmacophore_gate=True,
)

work_dir = './work_dirs/new_s0_reverse_kg_pharmacophore_p6_128_mean_gate'
