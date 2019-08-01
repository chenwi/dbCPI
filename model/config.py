import os
class Config:

    # model2 hyperparameters
    position_embedding_size = 35
    filter_size = 3
    num_filters = 1800
    gamma = 2.0
    m_plus = 2.5
    m_minus = 0.5
    l2_reg_lambda = 1.0e-4
    learning_rate = 0.002

    # training parameters
    batch_size = 16
    num_epochs = 8
    show_step = 10
    evaluate_every = 10000
    checkpoint_every = 10000

    # misc parameters
    allow_soft_placement = False
    log_device_placement = False

    # data dir
    data_dir = r'./data'
    log_file='run.log'
    checkpoint_dir = os.path.join('pretrained','checkpoints', 'model')
    train_set_path = os.path.join(data_dir, r'zipdata/train_set.pkl')

