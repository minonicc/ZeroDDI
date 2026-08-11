_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(
        use_sub=True,
        use_query_substructure=False,
        use_fixed_substructure_base=True,
        fixed_substructure_alpha_init=0.0,
    ),
    matching_use_substructure_evidence=False,
    semantic_aux_lambda=0.0,
)

# This debug-ready configuration uses P0 temporarily. Replace its pharmacophore
# controls with the stage-one winner before any formal Struct-S3 run.
work_dir = './work_dirs/new_s0_struct_s3_fixed_substructure_pharmacophore'
