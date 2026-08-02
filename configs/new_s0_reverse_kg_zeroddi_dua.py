_base_ = './new_s0_reverse_kg.py'

model = dict(
    class_balanced_beta=0.0,
    static_ddie_uniformity_lambda=0.0,
    zeroddi_dua_aux_lambda=0.3,
)

work_dir = './work_dirs/new_s0_reverse_kg_zeroddi_dua'
