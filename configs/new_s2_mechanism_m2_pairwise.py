_base_ = './new_s2_mechanism_m1_experts.py'

# M2 on S2: retain the three mechanism experts and add explicit pairwise
# interactions, while keeping candidate-conditioned pairwise gates disabled.
model = dict(
    matching_use_pairwise_interactions=True,
    matching_use_pairwise_gates=False,
)

work_dir = './work_dirs/new_s2_mechanism_m2_pairwise'
