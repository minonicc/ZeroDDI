_base_ = './new_s0_reverse_kg_pharmacophore.py'

# Direct P8-controlled filtering comparison: expose the complete valid pair
# set, rank it independently for each candidate DDIE, and retain 256 pairs.
model = dict(
    leftmodel=dict(pharmacophore_max_pairs=None),
    matching_pharmacophore_top_k=256,
    matching_pharmacophore_candidate_chunk_size=4,
    matching_use_pharmacophore_gate=False,
)

work_dir = './work_dirs/new_s0_pharmacophore_t2_top256'
