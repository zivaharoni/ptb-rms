class SmallConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 100
    time_steps = 35
    units_num = 650
    embedding_size = 650
    vocab_size = 10000
    lstm_layers_num = 3
    seed = 570164

    opt = "marms"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    opt_clip_by_var = None
    opt_mom = 0.0
    opt_mom_decay = 1.0
    opt_c_lipsc = 1.0

    lr = 1.0
    lr_decay = 0.9
    max_grad_norm = 0.0
    max_update_norm = 1.0
    layer_epoch = 500
    entire_network_epoch = layer_epoch

    GL = True
    DC = False
    AR = 1.0
    TAR = 2.0
    variational = 'epoch'
    keep_prob_embed = 0.35
    drop_output = [0.65,0.3]
    drop_state = [0.65,0.65]
    drop_input = 0.9
    drop_embed_var = False

class BigConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 100
    time_steps = 35
    units_num = 1000
    embedding_size = 1000
    vocab_size = 10000
    lstm_layers_num = 4
    seed = 570164

    opt = "marms"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    opt_clip_by_var = None
    opt_mom = 0.0
    opt_mom_decay = 1.0
    opt_c_lipsc = 1.0

    lr = 1.0
    lr_decay = 0.9
    max_grad_norm = 0.0
    max_update_norm = 1.0
    layer_epoch = 500
    entire_network_epoch = layer_epoch

    GL = True
    DC = False
    AR = 1.0
    TAR = 2.0
    variational = 'epoch'
    keep_prob_embed = 0.3
    drop_output = [0.55, 0.25]
    drop_state = [0.55, 0.55]


class BiggerConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 100
    time_steps = 35
    units_num = 1200
    embedding_size = 1200
    vocab_size = 10000
    lstm_layers_num = 2
    seed = 570164

    opt = "marms"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    opt_clip_by_var = None
    opt_mom = 0.0
    opt_mom_decay = 1.0

    lr = 1.0
    lr_decay = 0.9
    max_grad_norm = 0.0
    max_update_norm = 2.5
    layer_epoch = 500
    entire_network_epoch = layer_epoch

    GL = True
    DC = False
    AR = 1.0
    TAR = 2.0
    variational = 'epoch'
    keep_prob_embed = 0.35
    drop_output = [0.65,0.25]
    drop_state = [0.65,0.25]


class TestConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 100
    time_steps = 35
    units_num = 650
    embedding_size = 650
    vocab_size = 10000
    lstm_layers_num = 1
    seed = 570164

    opt = "arms"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    opt_clip_by_var = None
    opt_mom = 0.0
    opt_mom_decay = 1.0

    lr = 1.0
    lr_decay = 0.9
    max_grad_norm = 0.0
    max_update_norm = 7.0
    layer_epoch = 50
    entire_network_epoch = layer_epoch

    GL = True
    DC = False
    AR = 1.0
    TAR = 2.0
    variational = 'epoch'
    keep_prob_embed = 0.5
    drop_output = [0.65, 0.35]
    drop_state = [0.65, 0.35]




class BestConfig(object):
    init_scale = 0.0258198889747
    lr = 1.0
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 35
    batch_size = 80
    units_num = 1500
    embedding_size = 1500
    vocab_size = 10000
    layer_epoch = 120
    entire_network_epoch = layer_epoch
    forget_bias_init = 0.0
    lstm_layers_num = 4
    GL = True
    DC = False
    AR = 2.0
    TAR = 1.0
    variational = 'epoch'
    opt = "sgd"
    keep_prob_embed = 0.3
    drop_output = [[0.3, 0.0, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0], [0.5, 0.5, 0.25, 0.0], [0.5, 0.5, 0.5, 0.25]]
    drop_state = [[0.3, 0.0, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0], [0.5, 0.5, 0.25, 0.0], [0.5, 0.5, 0.5, 0.25]]

    cache_norm = None
    cache_size = 2000
    cache_alpha = 0
    cache_lambd = 0.1
    cache_theta = 5.

    dynamic_eval = False
    dynamic_rms_step = None
    dynamic_rms_decay = None
    dynamic_decay = 2e-3
    dynamic_lr = 5e-5
    dynamic_time_steps = 5
    dynamic_epsilon = 1e-3
    max_update_norm = 1.
    dynamic_clip_total_update = None

# python model_estimator.py --model small --start_layer 2 --ckpt_file ./results/small/saver/best_model1 --dynamic_eval True --dynamic_time_steps 5 --dynamic_lr 0.0001 --dynamic_decay 1e-05 --dynamic_rms_step True --dynamic_rms_decay True