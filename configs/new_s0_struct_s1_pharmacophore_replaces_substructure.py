_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(
        use_sub=False,
        use_query_substructure=False,
        use_fixed_substructure_base=False,
    ),
    matching_use_substructure_evidence=False,
    semantic_aux_lambda=0.0,
)

work_dir = './work_dirs/new_s0_struct_s1_pharmacophore_replaces_substructure'
