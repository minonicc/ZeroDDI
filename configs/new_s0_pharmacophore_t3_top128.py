_base_ = './new_s0_reverse_kg_pharmacophore.py'

# Direct P0-controlled filtering comparison: rank the complete valid pair set
# independently for every candidate DDIE, retain 128 pairs, then renormalize
# attention.  This changes selection only versus P0's fixed first-128 prefix.
model = dict(
    leftmodel=dict(pharmacophore_max_pairs=None),
    matching_pharmacophore_top_k=128,
    # Memory-only execution control: process four candidate DDIEs at a time.
    # This does not change scores, Top-K membership, or aggregation semantics.
    matching_pharmacophore_candidate_chunk_size=4,
)

work_dir = './work_dirs/new_s0_pharmacophore_t3_top128'
