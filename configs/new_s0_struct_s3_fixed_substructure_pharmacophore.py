_base_ = './new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py'

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

# P3 (128 + sum + gate) was selected by the stage-one 50-epoch validation
# screen.  Struct-S3 changes only the substructure role relative to that winner.
work_dir = './work_dirs/new_s0_struct_s3_fixed_substructure_pharmacophore'
