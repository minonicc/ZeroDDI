_base_ = './new_s0_pharmacophore_n1_nodes.py'

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

work_dir = './work_dirs/new_s0_kg_gnn_pharmacophore_n1'
train_batch_size = 64

# Keep the final N1 node-attention architecture and bound CPU parallelism.
cpu_threads = 8
detect_anomaly = False
