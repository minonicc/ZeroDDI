_base_ = './new_s0_reverse_kg_pharmacophore.py'

# P8-controlled single-drug filtering: select each side's candidate-specific
# Top-16 pharmacophores, then construct and aggregate at most 256 real pairs.
model = dict(
    leftmodel=dict(
        pharmacophore_selection_mode='drug_topk',
        pharmacophore_max_pairs=None,
    ),
    matching_pharmacophore_drug_top_k=16,
    matching_pharmacophore_candidate_chunk_size=2,
    matching_use_pharmacophore_gate=False,
)

work_dir = './work_dirs/new_s0_pharmacophore_d3_drug_top16'
