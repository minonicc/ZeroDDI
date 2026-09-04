_base_ = './new_s1_pharmacophore_n1_nodes.py'

# Transfer the complete M3 architecture from S0 to S1 without changing its
# mechanism experts, pairwise interactions, or candidate-conditioned gates.
model = dict(
    matching_use_mechanism_experts=True,
    matching_use_pairwise_interactions=True,
    matching_use_pairwise_gates=True,
)

work_dir = './work_dirs/new_s1_mechanism_m3_pairwise_gates'
