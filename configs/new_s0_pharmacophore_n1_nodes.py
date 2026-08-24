_base_ = './new_s0_reverse_kg_pharmacophore.py'

# Strict substructure analogue: expose both drugs' typed pharmacophore nodes,
# concatenate them into one evidence set, and let every candidate DDIE query
# that set directly. No pharmacophore pairs or hard Top-K are constructed.
model = dict(
    leftmodel=dict(
        pharmacophore_selection_mode='drug_nodes',
        pharmacophore_max_pairs=None,
    ),
    matching_pharmacophore_use_drug_nodes=True,
    matching_pharmacophore_top_k=None,
    matching_pharmacophore_drug_top_k=None,
    matching_pharmacophore_use_null_evidence=True,
    matching_use_pharmacophore_gate=False,
)

work_dir = './work_dirs/new_s0_pharmacophore_n1_nodes'
