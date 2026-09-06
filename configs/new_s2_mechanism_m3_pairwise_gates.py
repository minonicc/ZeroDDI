_base_ = './new_s2_pharmacophore_n1_nodes.py'

# Transfer the complete M3 architecture to S2 without changing its mechanism
# experts, explicit pairwise interactions, or candidate-conditioned gates.
model = dict(
    matching_use_mechanism_experts=True,
    matching_use_pairwise_interactions=True,
    matching_use_pairwise_gates=True,
)

work_dir = './work_dirs/new_s2_mechanism_m3_pairwise_gates'
