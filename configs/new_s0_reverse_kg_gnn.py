_base_ = './new_s0_reverse_kg.py'

kg_pair_file = './data/KnowDDI/drugbank_true_s0/kg/molecbionet_pair_graph_s0.sqlite'

model = dict(
    matching_kg_graph_evidence=True,
    matching_kg_evidence_dim=300,
    matching_kg_hidden_dim=256,
)

data = dict(
    train=dict(kg_pair_file=kg_pair_file, kg_max_nodes=256, kg_max_edges=1024),
    zsl_val=dict(kg_pair_file=kg_pair_file, kg_max_nodes=256, kg_max_edges=1024),
    gzsl_val=dict(kg_pair_file=kg_pair_file, kg_max_nodes=256, kg_max_edges=1024),
    zsl_test=dict(kg_pair_file=kg_pair_file, kg_max_nodes=256, kg_max_edges=1024),
    gzsl_test=dict(kg_pair_file=kg_pair_file, kg_max_nodes=256, kg_max_edges=1024),
)

work_dir = './work_dirs/new_s0_reverse_kg_gnn'
train_batch_size = 64
