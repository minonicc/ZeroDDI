_base_ = './new_s0_mechanism_m2_pairwise.py'

# M3: make each pairwise interaction candidate-conditioned. A two-way softmax
# allocates relative weight to the two participating mechanism experts.
model = dict(
    matching_use_pairwise_gates=True,
)

work_dir = './work_dirs/new_s0_mechanism_m3_pairwise_gates'
