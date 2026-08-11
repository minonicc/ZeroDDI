_base_ = './new_s0_reverse_kg.py'

model = dict(
    leftmodel=dict(
        use_sub=True,
        use_query_substructure=False,
        use_fixed_substructure_base=True,
        fixed_substructure_alpha_init=0.0,
    ),
    matching_use_substructure_evidence=False,
    matching_use_pharmacophore_evidence=False,
    semantic_aux_lambda=0.0,
)

work_dir = './work_dirs/new_s0_struct_s4_fixed_substructure'
