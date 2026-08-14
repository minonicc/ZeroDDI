_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(
        use_sub=False,
        use_query_substructure=False,
        use_fixed_substructure_base=False,
    ),
    matching_use_substructure_evidence=False,
    matching_use_pharmacophore_gate=False,
    semantic_aux_lambda=0.0,
)

# Hold the selected P0 controls (128 + sum + no gate) fixed so this ablation
# changes only the role of the DDIE-query substructure branch.
work_dir = './work_dirs/new_s0_struct_s1_pharmacophore_replaces_substructure'
