_base_ = './new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py'

provisional_dependency = 'stage1_formal100_validation_winner'

model = dict(
    leftmodel=dict(
        use_sub=False,
        use_query_substructure=False,
        use_fixed_substructure_base=False,
    ),
    matching_use_substructure_evidence=False,
    semantic_aux_lambda=0.0,
)

# Hold the stage-one P3 pharmacophore controls fixed so this ablation changes
# only the role of the DDIE-query substructure branch.
work_dir = './work_dirs/new_s0_struct_s1_pharmacophore_replaces_substructure'
