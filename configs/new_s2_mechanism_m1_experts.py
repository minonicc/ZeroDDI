_base_ = './new_s2_pharmacophore_n1_nodes.py'

model = dict(
    matching_use_mechanism_experts=True,
    matching_use_pairwise_interactions=False,
    matching_use_pairwise_gates=False,
)

work_dir = './work_dirs/new_s2_mechanism_m1_experts'
