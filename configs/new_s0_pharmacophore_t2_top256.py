_base_ = './new_s0_reverse_kg_pharmacophore.py'

provisional_dependency = 'stage2_validation_winner'

# Engineering template only; formal inheritance waits for stage two.
model = dict(
    leftmodel=dict(pharmacophore_max_pairs=None),
    matching_pharmacophore_top_k=256,
)

work_dir = './work_dirs/new_s0_pharmacophore_t2_top256'
