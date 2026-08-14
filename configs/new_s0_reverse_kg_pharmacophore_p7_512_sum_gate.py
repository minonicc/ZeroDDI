_base_ = './new_s0_reverse_kg_pharmacophore_p2_512_sum.py'

model = dict(
    matching_use_pharmacophore_gate=True,
)

work_dir = './work_dirs/new_s0_reverse_kg_pharmacophore_p7_512_sum_gate'
