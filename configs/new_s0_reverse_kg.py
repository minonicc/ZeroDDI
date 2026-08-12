_base_ = './zeroddi_reverse_kg_seen.py'

knowddi_all_file = './data/KnowDDI/drugbank_true_s0/DDI_final1.5.csv'
knowddi_split_dir = './data/KnowDDI/drugbank_true_s0'
knowddi_output_dir = './data/KnowDDI/output_s0/'
kg_pair_file = './data/KnowDDI/drugbank_true_s0/kg/molecbionet_pair_kg_s0.json'

model = dict(
    leftmodel=dict(Allfilename=knowddi_all_file),
)

data = dict(
    train=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='train.csv',
        output_file=knowddi_output_dir,
        kg_pair_file=kg_pair_file,
        kg_max_tokens=128,
    ),
    zsl_val=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='val.csv',
        output_file=knowddi_output_dir,
        kg_pair_file=kg_pair_file,
        kg_max_tokens=128,
    ),
    gzsl_val=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='test.csv',
        output_file=knowddi_output_dir,
        kg_pair_file=kg_pair_file,
        kg_max_tokens=128,
    ),
    zsl_test=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='test.csv',
        output_file=knowddi_output_dir,
        kg_pair_file=kg_pair_file,
        kg_max_tokens=128,
    ),
    gzsl_test=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='test.csv',
        output_file=knowddi_output_dir,
        kg_pair_file=kg_pair_file,
        kg_max_tokens=128,
    ),
    val_seen=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='val.csv',
        output_file=knowddi_output_dir,
    ),
    test_seen=dict(
        Allfilename=knowddi_all_file,
        file_dir=knowddi_split_dir,
        file_name='test.csv',
        output_file=knowddi_output_dir,
    ),
)

selection_metric = 'Macro-F1'
train_batch_size = 128
cpu_threads = 8
work_dir = './work_dirs/new_s0_reverse_kg'
