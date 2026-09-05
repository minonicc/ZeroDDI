_base_ = './new_s1_pharmacophore_n1_nodes.py'

# Transfer M1 unchanged from S0 to S1: three mechanism-specific experts, with
# no pairwise interaction modules and no pairwise gates.
model = dict(
    matching_use_mechanism_experts=True,
    matching_use_pairwise_interactions=False,
    matching_use_pairwise_gates=False,
)

work_dir = './work_dirs/new_s1_mechanism_m1_experts'
