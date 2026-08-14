_base_ = './new_s0_pharmacophore_t3_top128.py'

# Candidate-specific Top-128 remains unchanged. Unlike softmax, each selected
# pair receives an independent sigmoid relevance weight. Dividing by the valid
# pair count (not by the weight sum) preserves the absolute relevance scale, so
# an all-irrelevant set can contribute evidence close to zero.
model = dict(
    matching_pharmacophore_top_k_aggregation='sigmoid_mean',
    matching_pharmacophore_use_null_evidence=False,
)

work_dir = './work_dirs/new_s0_pharmacophore_t3_top128_sigmoid_mean'
