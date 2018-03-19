class SmallConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 200
    time_steps = 35
    units_num = [650, 650, 650]
    embedding_size = 650
    vocab_size = 10000
    lstm_layers_num = 3
    seed = 570164

    opt = "marms"
    opt_eps = 1e-5
    opt_inverse_type = "add"

    lr = 1.0
    lr_decay = 1.0
    max_update_norm = 1.0
    layer_epoch = 500
    entire_network_epoch = layer_epoch

    GL = True
    DC = False
    AR = 1.0
    TAR = 2.0
    variational = 'epoch'
    keep_prob_embed = 0.35
    drop_output = [0.65,0.30]
    drop_state = [0.65,0.65]
    drop_i = 1.0
    drop_embed_var = False

    mos = False
    mos_context_num = 0
    mos_drop = 0.0

class MosConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 12
    time_steps = 70
    units_num = [960, 960, 620]
    embedding_size = 280
    vocab_size = 10000
    lstm_layers_num = 3
    seed = 570164

    opt = "asgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"

    lr = 20.0
    lr_decay = 1.0
    max_update_norm = 0.25
    layer_epoch = 1000
    entire_network_epoch = layer_epoch

    GL = False
    DC = False
    AR = 2.0
    TAR = 1.0
    variational = 'epoch'
    keep_prob_embed = 0.6
    drop_output = [0.775,0.6]
    drop_state = [0.5,0.5]
    drop_i = 0.9
    drop_embed_var = False
    wdecay = 1.2e-6

    mos = True
    mos_context_num = 15
    mos_drop = 0.71


class MosGLConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 12
    time_steps = 70
    units_num = [850, 850, 850]
    embedding_size = 280
    vocab_size = 10000
    lstm_layers_num = 3
    seed = 570164

    opt = "asgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"

    lr = 20.0
    lr_decay = 1.0
    max_update_norm = 0.25
    layer_epoch = 200
    entire_network_epoch = layer_epoch

    GL = True
    DC = False
    AR = 2.0
    TAR = None
    variational = 'epoch'
    keep_prob_embed = 0.6
    drop_output = [0.775,0.6]
    drop_state = [0.5,0.5]
    drop_i = 0.9
    drop_embed_var = False
    wdecay = 1.2e-6

    mos = True
    mos_context_num = 15
    mos_drop = 0.71

class BestConfig(object):
    init_scale = 0.0258198889747
    time_steps = 35
    batch_size = 20
    units_num = [1500] * 3
    embedding_size = 1500
    vocab_size = 10000
    seed = 570164

    opt = "asgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    lr = 1.0
    lr_decay = 0.85
    max_update_norm = 5.0
    layer_epoch = 300
    entire_network_epoch = layer_epoch
    forget_bias_init = 0.0
    lstm_layers_num = 3
    GL = True #TODO cahnge to true
    DC = False
    AR = 2.0
    TAR = 1.0

    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [0.5, 0.27]
    drop_state = [0.5, 0.27]
    drop_embed_var = False
    drop_i = 1.0

    mos = False
    mos_context_num = 0
    mos_drop = 0.0


# python model_estimator.py --model small --start_layer 2 --ckpt_file ./results/small/saver/best_model1 --dynamic_eval True --dynamic_time_steps 5 --dynamic_lr 0.0001 --dynamic_decay 1e-05 --dynamic_rms_step True --dynamic_rms_decay True

