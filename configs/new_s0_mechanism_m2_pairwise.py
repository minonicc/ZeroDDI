_base_ = './new_s0_mechanism_m1_experts.py'

# M2: explicitly model S-K, S-P, and K-P interactions. Every interaction uses
# the two expert vectors, their elementwise product, and absolute difference.
model = dict(
    matching_use_pairwise_interactions=True,
    matching_use_pairwise_gates=False,
)

work_dir = './work_dirs/new_s0_mechanism_m2_pairwise'
