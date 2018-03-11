from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import ptb_reader
import ptb_config
import os
import re
import sys
import ast
import rnn_cell_additions as dr
from dynamic_eval import DynamicEval

flags = tf.flags
logging = tf.logging

# flags for distributed tf
flags.DEFINE_string(
    "gpu_devices", "/gpu:0", "list gpu devices")
flags.DEFINE_string(
    "cpu_device", "/cpu:0", "name cpu device")

# flags for error handling
flags.DEFINE_string(
    "ckpt_file", None, "name of checkpoint file to load model from")
flags.DEFINE_integer(
    "start_layer", None, "if restore is needed, mention the layer to start from")

# flags for model
flags.DEFINE_bool(
    "dynamic_ms_norm", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_bool(
    "dynamic_clip_total_update", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_bool(
    "dynamic_eval", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "dynamic_decay", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "dynamic_epsilon", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "max_update_norm", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "max_grad_norm", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "dynamic_lr", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "dynamic_eps", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_integer(
    "dynamic_time_steps", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_bool(
    "dynamic_rms_decay", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_bool(
    "dynamic_rms_step", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_string(
    "cache_norm", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "cache_theta", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_float(
    "cache_lambda", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_integer(
    "cache_size", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_string(
    "name", None, "model name. describe shortly the purpose of the run")
flags.DEFINE_string(
    "model", "best", "model type. options are: GL, LAD, GL_LAD, Deep_GL_LAD, test")
flags.DEFINE_string(
    "data", "ptb", "dataset. options are: ptb, enwik8")
flags.DEFINE_integer(
    "seed", 0, "random seed used for initialization")
flags.DEFINE_integer(
    "batch_size", None, "#of sequences for all gpus in total")
flags.DEFINE_integer(
    "lstm_layer_num", None, "#of lstm layers")
flags.DEFINE_integer(
    "layer_epoch", None, "#of epochs per layer")
flags.DEFINE_float(
    "lr", None, "initial lr")
flags.DEFINE_float(
    "lr_decay", None, "lr decay when validation decreases")
flags.DEFINE_integer(
    "time_steps", None, "bptt truncation")
flags.DEFINE_bool(
    "GL", None, "gradual learning of the network")
flags.DEFINE_bool(
    "DC", None, "drop connect lstm's hidden-to-hidden connections")
flags.DEFINE_float(
    "AR", None, "activation regularization coefficient")
flags.DEFINE_float(
    "TAR", None, "temporal activation regularization coefficient")
flags.DEFINE_string(
    "opt", None, "sgd or asgd optimizer")
flags.DEFINE_integer(
    "embedding_size", None, "#of units in the embedding representation")
flags.DEFINE_integer(
    "units_num", None, "#of units in lstm cell")
flags.DEFINE_float(
    "keep_prob_embed", None, "keep prob of embedding representation unit")
flags.DEFINE_string(
    "drop_output", None, "keep prob of lstm output")
flags.DEFINE_string(
    "drop_state", None, "keep prob of lstm state")
FLAGS = flags.FLAGS


class PTBModel(object):
    """class for handling the ptb model"""

    def __init__(self,
                 config,
                 is_training,
                 inputs):
        """the constructor builds the tensorflow graph"""
        self._input = inputs
        vocab_size = config.vocab_size  # num of possible words
        self._gpu_devices = [i for i in range(len(get_gpu_devices(FLAGS.gpu_devices)))]
        self._gpu_num = len(self._gpu_devices)
        self._cpu_device = FLAGS.cpu_device

        with tf.name_scope("model_variables"):
            with tf.name_scope("global_step"):
                self._global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.name_scope("epoch_counter"):
                self._epoch_count = tf.Variable(0, name='epoch', trainable=False)
                self._epoch_inc = tf.assign(self._epoch_count, tf.add(self._epoch_count, tf.constant(1)))
                self._epoch_reset = tf.assign(self._epoch_count, tf.constant(0))

        # ptrs to the lstm cell object, ltsm initial state op and final state
        self._cell = []
        self._initial_state = []
        self._final_state = []

        # construct the embedding layer on cpu device
        with tf.variable_scope("embedding"), tf.device(self._cpu_device):
            # the embedding matrix is allocated in the cpu to save valuable gpu memory for the model.
            embedding_map = tf.get_variable(
                name="embedding", shape=[vocab_size, config.embedding_size], dtype=tf.float32)
            b_embed_in = tf.get_variable(name="b_embed_in", shape=[config.embedding_size], dtype=tf.float32)
            embedding = tf.nn.embedding_lookup(embedding_map, self._input.input_data) + b_embed_in

            # non variational wrapper for the embedding
            if is_training and config.keep_prob_embed < 1:
                embedding_out = tf.nn.dropout(embedding,
                                              config.keep_prob_embed)  # / config.keep_prob_embed
            else:
                embedding_out = embedding

        # split input to devices if needed
        with tf.name_scope("split_inputs"):
            if self._gpu_num > 1:
                embedding_out = tf.split(embedding_out, self._gpu_num)
                targets = tf.split(inputs.targets, self._gpu_num)
            else:
                embedding_out = [embedding_out]
                targets = [inputs.targets]

        # construct the rest of the model on every gpu
        all_loss = []  # 2D array of scalar loss; [i,j] element stands for the loss of the j-th layer of the i-th gpu
        all_grads = []  # 2D array of grads; [i,j] element stands for the grad of the j-th layer of the i-th gpu

        with tf.variable_scope("gpus"):
            for i in range(self._gpu_num):
                with tf.device("/gpu:%d" % self._gpu_devices[i]), tf.name_scope("gpu-%d" % i):
                    loss, grads, cell, initial_state, final_state, cache_data = self.complete_model(embedding_out[i],
                                                                                                    embedding_map,
                                                                                                    config,
                                                                                                    is_training,
                                                                                                    inputs,
                                                                                                    targets[i])

                    self._cache_data = cache_data
                    self._cell.append(cell)
                    self._initial_state.append(initial_state)
                    self._final_state.append(final_state)
                    all_loss.append(loss)
                    all_grads.append(grads)

                    # reuse variables for the next gpu
                    tf.get_variable_scope().reuse_variables()

        # reduce per-gpu-loss to total loss
        with tf.name_scope("reduce_loss"):
            self._loss = self.reduce_loss(all_loss)

        if config.dynamic_eval is not None:
            # average grads ; sync point
            with tf.name_scope("average_grads"):
                averaged_grads = self.average_grads(all_grads)

            # get trainable vars
            tvars = tf.trainable_variables()

            self._dynamic_eval = DynamicEval(config, tvars, averaged_grads)

            self._train_op = self._dynamic_eval.update_op()

    def reduce_loss(self, all_loss):
        """ average the loss obtained by gpus

        Args:
            all_loss: 2D array, the [i,j] element stands for the loss of the j-th layer of the i-th gpu

        Returns:
            total_loss: a list of the loss for each layer
        """
        if self._gpu_num == 1:
            total_loss = all_loss[0]
        else:
            layer_loss = [all_loss[j] for j in range(self._gpu_num)]
            total_loss = tf.reduce_mean(layer_loss)

        return total_loss

    def average_grads(self, all_grads):
        """ average the grads of the currently trained layer

        Args:
            grads: 2D array, the [i,j] element stands for the loss of the j-th layer of the i-th gpu

        Returns:
            grads: a list of the averaged grads for each layer
        """

        if self._gpu_num == 1:
            average_layer_grads = all_grads[0]
        else:
            layer_grads = [all_grads[i][-1][:] for i in range(len(all_grads))]
            grads = []
            for grad_and_vars in zip(*layer_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                gpu_grads = []

                for g in grad_and_vars:
                    if g is not None:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        gpu_grads.append(expanded_g)

                if g is not None:
                    # Average over the 'tower' dimension.
                    grad = tf.concat(axis=0, values=gpu_grads)
                    grad = tf.reduce_mean(grad, 0)
                else:
                    grad = g

                grads.append(grad)

            average_layer_grads = grads
        return average_layer_grads

    def complete_model(self, embedding_out, embedding_map, config, is_training, inputs, targets):
        """ Build rest of model for a single gpu

        Args:
            embedding_out: the embedding representation to be processed

        Returns:
            loss: a list for the loss calculated for each layer.
            grads: a list for the grads calculated for each loss.
        """

        batch_size = inputs.batch_size // self._gpu_num  # num of sequences
        assert inputs.batch_size // self._gpu_num == inputs.batch_size / self._gpu_num, \
            "must choose batch size that is divided by gpu_num"

        time_steps = config.time_steps  # num of time steps used in BPTT
        vocab_size = config.vocab_size  # num of possible words
        units_num = config.units_num  # num of units in the hidden layer

        # define basic lstm cell
        def lstm_cell(lstm_size):
            if config.DC:
                return dr.WeightDroppedLSTMCell(num_units=lstm_size,
                                                is_training=is_training,
                                                state_is_tuple=True)
            else:
                return tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size,
                                                    forget_bias=config.forget_bias_init,
                                                    state_is_tuple=True)

        possible_cell = lstm_cell
        # if dropout is needed add a dropout wrapper
        if is_training and config.drop_output is not None:
            def possible_cell(lstm_size):
                if config.variational is not None:
                    if config.DC:
                        return dr.WeightDroppedVariationalDropoutWrapper(lstm_cell(lstm_size),
                                                                         batch_size,
                                                                         lstm_size)
                    else:
                        return dr.VariationalDropoutWrapper(lstm_cell(lstm_size),
                                                            batch_size,
                                                            lstm_size)
                else:
                    return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size),
                                                         output_keep_prob=config.drop_output)

        # organize layers' outputs and states in a list
        cell = []
        initial_state = []
        outputs = []
        state = []
        lstm_output = []
        for _ in range(config.lstm_layers_num):
            outputs.append([])
            state.append([])

        # unroll the cell to "time_steps" times
        # first lstm layer
        with tf.variable_scope("lstm%d" % 1):
            lstm_size = units_num
            cell.append(possible_cell(lstm_size))
            initial_state.append(cell[0].zero_state(batch_size, dtype=tf.float32))
            state[0] = initial_state[0]
            for time_step in range(time_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (new_h, state[0]) = cell[0](embedding_out[:, time_step, :], state[0])
                outputs[0].append(new_h)
            lstm_output.append(tf.reshape(tf.concat(values=outputs[0], axis=1), [-1, lstm_size]))

        # rest of layers
        for i in range(1, config.lstm_layers_num):
            with tf.variable_scope("lstm%d" % (i + 1)):
                lstm_size = config.embedding_size if i == (config.lstm_layers_num - 1) else units_num
                cell.append(possible_cell(lstm_size))
                initial_state.append(cell[i].zero_state(batch_size, dtype=tf.float32))
                state[i] = initial_state[i]
                for time_step in range(time_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (new_h, state[i]) = cell[i](outputs[i - 1][time_step], state[i])
                    outputs[i].append(new_h)
                lstm_output.append(tf.reshape(tf.concat(values=outputs[i], axis=1), [-1, lstm_size]))

        # outer embedding bias
        b_embed_out = tf.get_variable(name="b_embed_out", shape=[vocab_size], dtype=tf.float32)
        # outer softmax matrix is tied with embedding matrix
        w_out = tf.transpose(embedding_map)

        # get trainable vars
        tvars = tf.trainable_variables()

        # since using GL we have logits, losses and cost for every layer

        with tf.name_scope("loss"):
            with tf.name_scope("data_loss"):
                logits = tf.matmul(lstm_output[-1], w_out) + b_embed_out
                exp_h = tf.exp(logits)
                h = lstm_output[-1]
                y = targets
                losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                            [tf.reshape(targets, [-1])],
                                                                            [tf.ones([batch_size * time_steps],
                                                                                     dtype=tf.float32)])
            if config.AR and is_training:
                with tf.name_scope("AR"):
                    for j in range(config.lstm_layers_num):
                        losses += config.AR * tf.reduce_mean(tf.square(tf.reshape(lstm_output[j], [-1, 1])))

            if config.TAR and is_training:
                with tf.name_scope("TAR"):
                    for j in range(config.lstm_layers_num):
                        outputs_reshaped = tf.reshape(lstm_output[j], [config.batch_size, config.time_steps, -1])
                        diff = outputs_reshaped[:, :-1, :] - outputs_reshaped[:, 1:, :]
                        losses += config.TAR * tf.reduce_mean(tf.square(tf.reshape(diff, [-1, 1])))

            loss = tf.reduce_sum(losses) / batch_size

            with tf.name_scope("compute_grads"):
                grads = tf.gradients(loss, tvars)

        final_state = state

        return loss, grads, cell, initial_state, final_state, (exp_h, h, y)

    def initial_state(self, device_num):
        return self._initial_state[device_num]

    def final_state(self, device_num):
        return self._final_state[device_num]

    @property
    def cache_data(self):
        return self._cache_data

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._optimizer.train_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def input(self):
        return self._input

    @property
    def lr(self):
        return self._lr

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch_count

    @property
    def gpu_num(self):
        return self._gpu_num

    @property
    def dynamic_eval(self):
        return self._dynamic_eval

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def epoch_inc(self, session):
        return session.run(self._epoch_inc)

    def epoch_reset(self, session):
        return session.run(self._epoch_reset)

    def gen_masks(self, session):
        feed_dict = {}
        for j in range(self._gpu_num):
            for i in range(config.lstm_layers_num):
                feed_dict.update(self._cell[j][i].gen_masks(session))
        return feed_dict

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        for j in range(self._gpu_num):
            for i in range(config.lstm_layers_num):
                print("layer %d: out %.2f, state %.2f" % (i+1, output_keep_prob[i], state_keep_prob[i]))
                self._cell[j][i].update_drop_params(session,
                                                    output_keep_prob[i],
                                                    state_keep_prob[i])


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.time_steps = time_steps = config.time_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // time_steps
        self.input_data, self.targets = ptb_reader.ptb_producer(
            data, batch_size, time_steps, name=name)


def print_tvars():
    tvars = tf.trainable_variables()
    nvars = 0
    for var in tvars[1:]:
        sh = var.get_shape().as_list()
        nvars += np.prod(sh)
        # print ('var: %s, size: [%s]' % (var.name,', '.join(map(str, sh))))
    print (nvars, ' total variables')


def run_epoch_test(session, model, verbose=False):
    """run the given model over its data"""
    # fetches = {"ms": model.dynamic_eval.global_ms()}
    # vals = session.run(fetches)
    # ms = vals["ms"]
    # s = np.sum(np.sqrt([x for x in ms]))
    # print(s)



    start_time = time.time()
    losses = 0.0
    iters = 0

    # zeros initial state for all devices
    state = []
    for k in range(model.gpu_num):
        state.append(session.run(model.initial_state(k)))

    # evaluate loss and final state for all devices
    fetches = {"loss": model.loss}

    if config.dynamic_eval:
        fetches["update_op"] = model.dynamic_eval.update_op()


    for k in range(model.gpu_num):
        fetches["final_state%d" % k] = model.final_state(k)

    for step in range(model.input.epoch_size):
        # pass states between time batches
        feed_dict = {}
        for i in range(model.gpu_num):
            gpu_state = model.initial_state(i)
            for j, (c, h) in enumerate(gpu_state):
                feed_dict[c] = state[i][j].c
                feed_dict[h] = state[i][j].h

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]

        for k in range(model.gpu_num):
            state[k] = vals["final_state%d" % k]

        losses += loss
        iters += model.input.time_steps

        if verbose and step % (model.input.epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f bits: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(losses / iters), np.log2(np.exp(losses / iters)),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(losses / iters)


def accumulate_grad_ms(session, model, verbose=True):
    """run the given model over its data"""
    start_time = time.time()
    iters = 0

    # zeros initial state for all devices
    state = []
    for k in range(model.gpu_num):
        state.append(session.run(model.initial_state(k)))

    # evaluate loss and final state for all devices
    fetches = {
        "ms_update": model.dynamic_eval.accu_global_ms()
    }
    for k in range(model.gpu_num):
        fetches["final_state%d" % k] = model.final_state(k)

    for step in range(model.input.epoch_size):
        # pass states between time batches
        feed_dict = {}
        for i in range(model.gpu_num):
            gpu_state = model.initial_state(i)
            for j, (c, h) in enumerate(gpu_state):
                feed_dict[c] = state[i][j].c
                feed_dict[h] = state[i][j].h

        vals = session.run(fetches, feed_dict)

        for k in range(model.gpu_num):
            state[k] = vals["final_state%d" % k]

        iters += model.input.time_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, iters * model.input.batch_size / (time.time() - start_time)))

    session.run(model.dynamic_eval.average_global_ms())
    # fetches = {"ms": model.dynamic_eval.global_ms()}
    # vals = session.run(fetches)
    # ms = vals["ms"]
    # print([np.sum(x) for x in ms])

    return


def read_flags(config, FLAGS):
    # assign flags into config
    flags_dict = FLAGS.__dict__['__flags']
    for key, val in flags_dict.items():
        if val is not None and key.strip() not in ["ckpt_file"]:
            if key.startswith("drop"):
                val = ast.literal_eval(val)
            if key == "layer_epoch":
                setattr(config, "entire_network_epoch", val)

            setattr(config, key, val)

    return config


def get_config(config_name):
    if config_name == "big":
        return config_pool.BigConfig()
    if config_name == "small":
        return config_pool.SmallConfig()
    elif config_name == "best":
        return config_pool.BestConfig()
    else:
        raise ValueError("Invalid model: %s", config_name)


def print_config(config):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    print('\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))


def get_gpu_devices(str):
    devices_num = re.findall("[0-9]", str)
    return devices_num


config_pool = ptb_config
reader = ptb_reader.ptb_raw_data

config = get_config(FLAGS.model)
config = read_flags(config, FLAGS)
config.batch_size = 1
config.time_steps = config.dynamic_time_steps

if FLAGS.data == 'ptb':
    config.vocab_size = 10000
elif FLAGS.data == 'wiki2':
    config.vocab_size = 33278
else:
    raise ValueError("invalid data-set name")

if FLAGS.start_layer is not None:
    config.lstm_layers_num = FLAGS.start_layer

data_path = "./data"

raw_data = reader(data_path, FLAGS.data)
train_data, valid_data, test_data, _ = raw_data
train_data = train_data[:100000]

print("\n\n", "cmd:", " ".join(sys.argv), "\n")

res_dir = '/'.join(FLAGS.ckpt_file.split('/')[:-1])


def main(_):

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")

            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=False, config=config, inputs=train_input)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, inputs=valid_input)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=config, data=test_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=config, inputs=test_input)

        saver = tf.train.Saver(var_list=tf.trainable_variables())
        ms_saver = tf.train.Saver(var_list=m.dynamic_eval.global_ms())

    sess_config = tf.ConfigProto(device_count={"CPU": 2},
                                 inter_op_parallelism_threads=2,
                                 intra_op_parallelism_threads=8)

    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = ",".join(get_gpu_devices(FLAGS.gpu_devices))

    with tf.Session(graph=graph, config=sess_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        session.run(tf.global_variables_initializer())

        start_time = time.time()

        if FLAGS.ckpt_file is not None:
            print("\nloading model from: " + FLAGS.ckpt_file)
            saver.restore(session, FLAGS.ckpt_file)
        else:
            raise ValueError("model must be provided")


        print_config(config)

        print("\nEstimating model with dynamic evaluation...")
        print("-" * 45)

        if config.dynamic_eval:
            print("\nsetting global weights.....")
            session.run(m.dynamic_eval.set_global_vars())

        if config.dynamic_eval and (config.dynamic_rms_decay or config.dynamic_rms_step):
            if not os.path.exists(res_dir + "/ms" + str(config.dynamic_time_steps) + ".index"):
                print("\naccumulating grads' ms over training data.....")
                accumulate_grad_ms(session, m)
                ms_saver.save(session, res_dir + "/ms" + str(config.dynamic_time_steps))
            else:
                print("\nusing cached MS_g.....")
                ms_saver.restore(session, res_dir + "/ms" + str(config.dynamic_time_steps))

        if config.dynamic_rms_decay is not None:
            session.run(m.dynamic_eval.norm_ms_grads())


        # a= session.run(m.dynamic_eval.global_ms())
        # for b in a:
        #     print(np.sqrt(np.min(b)), np.sqrt(np.max(b)), np.linalg.norm(b))
        # exit()
        print("\nevaluating test and validation.....\n")
        valid_perplexity = run_epoch_test(session, mvalid, verbose=True)
        print("Valid Perplexity: %.3f Bits: %.3f \n" % (valid_perplexity, np.log2(valid_perplexity)))

        test_perplexity = run_epoch_test(session, mtest, verbose=True)
        print("Test Perplexity: %.3f Bits: %.3f " % (test_perplexity, np.log2(test_perplexity)))

        end_time = time.time()
        elapsed = end_time - start_time
        print("Estimation took %02d:%02d:%02d " % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

        coord.request_stop()
        coord.join(threads)


        f = open(res_dir + "/results.txt", 'a')
        l = list()
        l.append(str(config.dynamic_eval))
        if config.dynamic_eval is not None:
            l.append(str(config.dynamic_time_steps))
            l.append(str(config.dynamic_lr))
            l.append(str(config.dynamic_decay))
            l.append(str(config.dynamic_rms_decay))
            l.append(str(config.dynamic_rms_step))
            l.append(str(config.max_grad_norm))
            l.append(str(config.dynamic_clip_total_update))
            l.append(str(config.max_update_norm))
            l.append(str(config.dynamic_epsilon))
            l.append(str(config.max_update_norm))
        l.append(str(valid_perplexity))
        l.append(str(test_perplexity))

        f.write(",".join(l) + "\n")
        f.close()

if __name__ == "__main__":
    tf.app.run()

# python model_estimator.py --model best --ckpt_file ./results/best/best_model3 --start_layer 4 --dynamic_eval True --dynamic_lr 9e-2 --dynamic_decay 2.5e-3 --dynamic_time_steps 5 --dynamic_rms_decay True --dynamic_rms_step True --dynamic_rms_decay True --max_grad_norm 100. --max_update_norm 50. --dynamic_clip_total_update True --dynamic_epsilon 1e-10