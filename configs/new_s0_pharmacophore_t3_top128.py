_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(pharmacophore_max_pairs=None),
    matching_pharmacophore_top_k=128,
)

work_dir = './work_dirs/new_s0_pharmacophore_t3_top128'
