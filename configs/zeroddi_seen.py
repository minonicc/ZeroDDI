_base_ = './zeroddi.py'

eval_modes = ['seen']
eval_interval = 1
work_dir = './work_dirs/zeroddi_seen'

data = dict(
    train=dict(
        file_dir='./data/DrugBank5.1.9/seen_random',
        file_name='train.csv',
        zsl_mode='train',
    ),
    zsl_val=dict(
        file_dir='./data/DrugBank5.1.9/seen_random',
        file_name='val.csv',
        zsl_mode='zsl',
    ),
    gzsl_val=dict(
        file_dir='./data/DrugBank5.1.9/seen_random',
        file_name='test.csv',
        zsl_mode='gzsl',
    ),
    zsl_test=dict(
        file_dir='./data/DrugBank5.1.9/seen_random',
        file_name='test.csv',
        zsl_mode='zsl',
    ),
    gzsl_test=dict(
        file_dir='./data/DrugBank5.1.9/seen_random',
        file_name='test.csv',
        zsl_mode='gzsl',
    ),
)
