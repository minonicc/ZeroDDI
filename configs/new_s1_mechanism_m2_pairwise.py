_base_ = './new_s1_mechanism_m1_experts.py'

model = dict(
    matching_use_pairwise_interactions=True,
    matching_use_pairwise_gates=False,
)

work_dir = './work_dirs/new_s1_mechanism_m2_pairwise'
