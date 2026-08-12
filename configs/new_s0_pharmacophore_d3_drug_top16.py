_base_ = './new_s0_reverse_kg_pharmacophore.py'

# Engineering template only. Rebase onto the selected stage-two branch before
# a formal screen; keep its pharmacophore pooling, gate, and scorer unchanged.
model = dict(
    leftmodel=dict(
        pharmacophore_selection_mode='drug_topk',
        pharmacophore_max_pairs=None,
    ),
    matching_pharmacophore_drug_top_k=16,
)

work_dir = './work_dirs/new_s0_pharmacophore_d3_drug_top16'
