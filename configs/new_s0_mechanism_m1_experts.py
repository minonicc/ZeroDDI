_base_ = './new_s0_pharmacophore_n1_nodes.py'

# M1: each candidate-selected evidence source is interpreted by its own
# mechanism-specific expert. Raw selected evidence is replaced, not duplicated,
# in the final candidate scorer.
model = dict(
    matching_use_mechanism_experts=True,
    matching_use_pairwise_interactions=False,
    matching_use_pairwise_gates=False,
)

work_dir = './work_dirs/new_s0_mechanism_m1_experts'
