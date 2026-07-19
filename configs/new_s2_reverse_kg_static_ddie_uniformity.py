_base_ = './new_s2_reverse_kg.py'

model = dict(
    static_ddie_uniformity_lambda=0.1,
)

work_dir = './work_dirs/new_s2_reverse_kg_static_ddie_uniformity'
