_base_ = './new_s0_reverse_kg_pharmacophore.py'

# P0-budget two-stage filtering: select each side's candidate-specific Top-12
# pharmacophores, construct at most 144 pairs, then retain the candidate's best
# 128 real pairs before adding null evidence and applying softmax.
model = dict(
    leftmodel=dict(
        pharmacophore_selection_mode='drug_topk',
        pharmacophore_max_pairs=None,
    ),
    matching_pharmacophore_drug_top_k=12,
    matching_pharmacophore_drug_pair_top_k=128,
    matching_pharmacophore_candidate_chunk_size=2,
    matching_use_pharmacophore_gate=False,
)

work_dir = './work_dirs/new_s0_pharmacophore_d3_drug_top12'
