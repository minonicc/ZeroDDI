_base_ = './new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py'

model = dict(
    leftmodel=dict(
        use_sub=True,
        use_query_substructure=False,
        use_fixed_substructure_base=True,
        fixed_substructure_alpha_init=0.0,
        use_pharmacophore_pairs=False,
    ),
    matching_use_substructure_evidence=False,
    matching_use_pharmacophore_evidence=False,
    matching_use_pharmacophore_gate=False,
    semantic_aux_lambda=0.0,
)

# Struct-S3 and Struct-S4 share the selected P3 training controls.  This
# ablation disables only the pharmacophore branch on top of the same fixed
# substructure base.
work_dir = './work_dirs/new_s0_struct_s4_fixed_substructure'
