_base_ = './new_s0_reverse_kg_pharmacophore.py'

model = dict(
    leftmodel=dict(
        pharmacophore_selection_mode='drug_topk',
        pharmacophore_pooling='mean',
    ),
    matching_pharmacophore_drug_top_k=12,
)

work_dir = './work_dirs/new_s0_pharmacophore_d3_drug_top12'
