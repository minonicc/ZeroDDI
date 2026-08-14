_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(
        use_sub=True,
        use_query_substructure=False,
        use_fixed_substructure_base=True,
        fixed_substructure_alpha_init=0.0,
    ),
    matching_use_substructure_evidence=False,
    matching_use_pharmacophore_gate=False,
    semantic_aux_lambda=0.0,
)

# P0 (128 + sum + no gate) is the selected stage-one pharmacophore base under
# the user-specified S0 test structure-selection protocol. Struct-S3 changes
# only the substructure role relative to that base.
work_dir = './work_dirs/new_s0_struct_s3_fixed_substructure_pharmacophore'
