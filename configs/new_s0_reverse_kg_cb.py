_base_ = './new_s0_reverse_kg.py'

model = dict(
    class_balanced_beta=0.999,
)

work_dir = './work_dirs/new_s0_reverse_kg_cb'
