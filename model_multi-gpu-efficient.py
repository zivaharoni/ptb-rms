from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import numpy as np
import time
import ptb_reader
import ptb_config
import os
import sys
import ast
import re
import rnn_cell_additions as dr
import argparse
import logging
import shutil


class PTBModel(object):
    """class for handling the ptb model"""

    def __init__(self,
                 config,
                 is_training,
                 inputs):
        """the constructor builds the tensorflow graph"""
        self._input = inputs
        vocab_size = config.vocab_size  # num of possible words
        self._gpu_devices = [i for i in range(len(get_gpu_devices(args.gpu_devices)))][0]
        self._cpu_device = args.cpu_device
        self._config = config

        with tf.name_scope("model_variables"):
            with tf.name_scope("global_step"):
                self._global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.name_scope("epoch_counter"):
                self._epoch_count = tf.Variable(0, name='epoch', trainable=False)
                self._epoch_inc = tf.assign(self._epoch_count, tf.add(self._epoch_count, tf.constant(1)))
                self._epoch_reset = tf.assign(self._epoch_count, tf.constant(0))


        # construct the embedding layer on cpu device
        with tf.variable_scope("embedding"), tf.device(self._cpu_device):
            # the embedding matrix is allocated in the cpu to save valuable gpu memory for the model.
            embedding_map = tf.get_variable(
                name="embedding", shape=[vocab_size, config.embedding_size], dtype=tf.float32)
            b_embed_in = tf.get_variable(name="b_embed_in", shape=[config.embedding_size], dtype=tf.float32)
            embedding = tf.nn.embedding_lookup(embedding_map, self._input.input_data) + b_embed_in

            if is_training and config.keep_prob_embed < 1:
                # non variational wrapper for the embedding

                self._emb_mask = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, config.time_steps, config.embedding_size],
                                                   name="embedding_mask")
                if config.drop_embed_var:
                    with tf.name_scope("out_mask_gen"):
                        random_tensor = ops.convert_to_tensor(config.keep_prob_embed)
                        random_tensor += random_ops.random_uniform([config.batch_size, 1, config.embedding_size])
                        random_tensor = tf.tile(random_tensor, [1, config.time_steps, 1])
                        self._gen_emb_mask = math_ops.floor(random_tensor)
                else:
                    with tf.name_scope("out_mask_gen"):
                        random_tensor = ops.convert_to_tensor(config.keep_prob_embed)
                        random_tensor += random_ops.random_uniform([config.batch_size, config.time_steps, config.embedding_size])
                        self._gen_emb_mask = math_ops.floor(random_tensor)

                embedding_out = math_ops.div(embedding, config.drop_i*config.keep_prob_embed) * self._emb_mask

            else:
                embedding_out = embedding

        with tf.device("/gpu:%d" % self._gpu_devices), tf.name_scope("inner_model"):
            loss, grads, cell, initial_state, final_state, softmax = self.complete_model(embedding_out,
                                                                                         embedding_map,
                                                                                         is_training)

            self._softmax = softmax
            self._cell = cell
            self._initial_state = initial_state
            self._final_state = final_state
            self._loss = loss
            self._grads = grads

        if is_training:
            # set learning rate as variable in order to anneal it throughout training
            with tf.name_scope("learning_rate"):
                self._lr = tf.Variable(config.lr, trainable=False)
                # a placeholder to assign a new learning rate
                self._new_lr = tf.placeholder(
                    tf.float32, shape=[], name="new_learning_rate")

                # function to update learning rate
                self._lr_update = tf.assign(self._lr, self._new_lr)

            # get trainable vars
            tvars = tf.trainable_variables()

            # define an optimizer with the averaged gradients
            with tf.name_scope("optimizer"):
                self._optimizer = []
                if config.opt == "sgd":
                    self._optimizer = SGDOptimizer(self, grads, tvars)
                    self._train_op = self._optimizer.train_op
                elif config.opt == "asgd":
                    opt = SGDOptimizer(self, grads, tvars)
                    self._optimizer = ASGDOptimizer(self, opt.updates, tvars)
                    self._train_op = self._optimizer.train_op
                elif config.opt == "masgd":
                    opt = SGDOptimizer(self, grads, tvars)
                    self._optimizer = MASGDOptimizer(self, opt.updates, tvars)
                    self._train_op = self._optimizer.train_op
                elif config.opt == "rms":
                    self._optimizer = RMSpropOptimizer(self, grads, tvars)
                    self._train_op = self._optimizer.train_op
                elif config.opt == "arms":
                    opt = RMSpropOptimizer(self, grads, tvars)
                    self._optimizer = ASGDOptimizer(self, opt.updates, tvars)
                    self._train_op = self._optimizer.train_op
                elif config.opt == "marms":
                    opt = RMSpropOptimizer(self, grads, tvars)
                    self._optimizer = MASGDOptimizer(self, opt.updates, tvars)
                    self._train_op = self._optimizer.train_op
                else:
                    raise ValueError( config.opt + " is not a valid optimizer")

    def complete_model(self, embedding_out, embedding_map, is_training):
        """ Build rest of model for a single gpu

        Args:
            embedding_out: the embedding representation to be processed

        Returns:
            loss: a list for the loss calculated for each layer.
            grads: a list for the grads calculated for each loss.
        """
        targets = self._input.targets
        config = self._config
        batch_size = config.batch_size
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

        with tf.name_scope("loss"):
            with tf.name_scope("data_loss"):
                logits = tf.matmul(lstm_output[-1], w_out) + b_embed_out
                softmax = tf.nn.softmax(logits)
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
                grad = tf.gradients(loss, tf.trainable_variables())
                grads = grad

        final_state = state

        return loss, grads, cell, initial_state, final_state, softmax

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def softmax(self):
        return self._softmax

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def rms(self):
        return self._rms

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
    def config(self):
        return self._config

    @property
    def emb_mask(self):
        return self._emb_mask

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def epoch_inc(self, session):
        return session.run(self._epoch_inc)

    def epoch_reset(self, session):
        return session.run(self._epoch_reset)

    def gen_masks(self, session):
        feed_dict = {}
        for i in range(config.lstm_layers_num):
            feed_dict.update(self._cell[i].gen_masks(session))
        return feed_dict

    def gen_emb_mask(self, session):
        return {self._emb_mask: session.run(self._gen_emb_mask)}

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        for i in range(config.lstm_layers_num):
            if i < config.lstm_layers_num -1:
                logger.info("layer %d: out %.2f, state %.2f" % (i+1, output_keep_prob[0], state_keep_prob[0]))
                self._cell[i].update_drop_params(session,
                                                    output_keep_prob[0],
                                                    state_keep_prob[0])
            else:
                logger.info("layer %d: out %.2f, state %.2f" % (i + 1, output_keep_prob[1], state_keep_prob[1]))
                self._cell[i].update_drop_params(session,
                                                    output_keep_prob[1],
                                                    state_keep_prob[1])


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data):
        self.raw_data = data
        self.batch_size = batch_size = config.batch_size
        self.time_steps = time_steps = config.time_steps
        self.epoch_size = epoch_size =  (len(data)-1) // (batch_size * time_steps)
        self.data_len = data_len = epoch_size * batch_size * time_steps
        self.data = np.reshape(data[:data_len], newshape=[batch_size, time_steps*epoch_size])
        self.label = np.reshape(data[1:data_len+1], newshape=[batch_size, time_steps*epoch_size])
        self.start_idx = 1
        self.input_data = tf.placeholder(dtype=tf.int32,shape=[batch_size, time_steps], name="input")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, time_steps], name="targets")

    def shuffle(self):
        data = self.raw_data
        data_len = self.data_len
        batch_size = self.batch_size
        epoch_size =  self.epoch_size
        time_steps = self.time_steps
        self.start_idx = start_idx = np.random.randint(0, (len(data)-1) % (self.batch_size * self.time_steps))
        self.data = np.reshape(data[start_idx:start_idx + data_len], newshape=[batch_size, time_steps * epoch_size])
        self.label = np.reshape(data[1+start_idx:data_len + start_idx + 1], newshape=[batch_size, time_steps * epoch_size])

        logger.info("Batching from index %d" % self.start_idx)


    def get_batch(self, idx):
        return {self.input_data: self.data[:, idx:idx+self.time_steps],
                    self.targets: self.label[:, idx:idx + self.time_steps]}


class SGDOptimizer(object):
    def __init__(self, model, grads, tvars):

        self._updates = [model.lr * g for g in grads]
        optimizer = tf.train.GradientDescentOptimizer(1.0)

        if config.max_update_norm > 0:
            self._updates, _ = tf.clip_by_global_norm(self._updates, config.max_update_norm)

        self._train_op = optimizer.apply_gradients(
            zip(self._updates, tvars), global_step=model.global_step)

    @property
    def train_op(self):
        return self._train_op

    @property
    def updates(self):
        return self._updates


class MASGDOptimizer(object):

    count = 0

    def __init__(self, model, grads, tvars, decay=0.9999):
        optimizer = tf.train.GradientDescentOptimizer(1.0)

        self._model = model
        self._updates = grads

        self._train_op = list()
        self._train_op.append(optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step))

        self._save_vars = []
        self._load_vars = []
        self._final_vars = []
        self._final_assign_op = []
        for var in tvars:
            self._final_vars.append(tf.get_variable(var.op.name + "final%d" % type(self).count,
                                                    initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))
            with tf.name_scope("final_average"):
                cur_epoch_num = (tf.cast((model.epoch + 1) * model.input.epoch_size, dtype=tf.float32))
                self._final_assign_op.append(tf.assign(var, self._final_vars[-1] *
                                                       (1 - decay) / (1 - decay ** cur_epoch_num)))

            with tf.name_scope("assign_current_weights"):
                tmp_var = (tf.get_variable(var.op.name + "tmp%d" % type(self).count,
                                           initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))
                self._save_vars.append(tf.assign(tmp_var, var))
                self._load_vars.append(tf.assign(var, tmp_var))

        with tf.control_dependencies(self._train_op):
            for i, var in enumerate(tvars):
                self._train_op.append(tf.assign(self._final_vars[i], decay * self._final_vars[i] + var))

        type(self).count += 1

    @property
    def train_op(self):
        return self._train_op

    @property
    def final_assign_op(self):
        return self._final_assign_op

    @property
    def save_vars(self):
        return self._save_vars

    @property
    def load_vars(self):
        return self._load_vars


class ASGDOptimizer(object):

    count = 0

    def __init__(self, model, grads, tvars):

        optimizer = tf.train.GradientDescentOptimizer(1.0)

        self._model = model
        self._updates = grads

        with tf.name_scope("trigger"):
            self._trigger = tf.get_variable("ASGD_trigger%d" % type(self).count, initializer=tf.constant(False, dtype=tf.bool), trainable=False)
            self._set_trigger = tf.assign(self._trigger, True)

            self._T = tf.get_variable("T%d" % type(self).count, initializer=tf.constant(0, dtype=tf.int32), trainable=False)
            self._new_T = tf.placeholder(tf.int32, shape=[], name="new_T%d" % type(self).count)
            self._set_T = tf.assign(self._T, self._new_T)

        self._train_op = list()
        update_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
        self._train_op.append(update_op)

        self._save_vars = []
        self._load_vars = []
        self._final_vars = []
        self._final_assign_op = []
        for var in tvars:
            self._final_vars.append(tf.get_variable(var.op.name + "final%d" % type(self).count,
                                                    initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False)
                                    )
            with tf.name_scope("final_average"):
                cur_epoch_num = tf.cast((model.epoch - self._T + 1) * model.input.epoch_size, dtype=tf.float32)
                self._final_assign_op.append(tf.assign(var, self._final_vars[-1] / cur_epoch_num))

            with tf.name_scope("assign_current_weights"):
                tmp_var = (tf.get_variable(var.op.name + "tmp%d" % type(self).count,
                                           initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))
                self._save_vars.append(tf.assign(tmp_var, var))
                self._load_vars.append(tf.assign(var, tmp_var))

        def trigger_on():
            with tf.name_scope("trigger_is_on"):
                op = list()
                op.append(tf.identity(self._trigger))
                op.append(tf.identity(self._T))
                for i, var in enumerate(tvars):
                    op.append(tf.assign_add(self._final_vars[i], var))

            return op

        def trigger_off():
            with tf.name_scope("trigger_is_off"):
                op = list()
                op.append(tf.identity(self._trigger))
                op.append(tf.identity(self._T))
                for i, var in enumerate(tvars):
                    op.append(tf.identity(self._final_vars[i]))

            return op

        with tf. control_dependencies([update_op]):
            with tf.name_scope("trigger_mux"):
                self._train_op.append(tf.cond(self._trigger, lambda: trigger_on(), lambda: trigger_off()))


        type(self).count += 1

    def set_trigger(self, session):
        return session.run(self._set_trigger)

    @property
    def train_op(self):
        return self._train_op

    @property
    def trigger(self):
        return self._trigger

    def set_T(self, session, T):
        return session.run(self._set_T, feed_dict={self._new_T: T})

    @property
    def final_assign_op(self):
        return self._final_assign_op

    @property
    def save_vars(self):
        return self._save_vars

    @property
    def load_vars(self):
        return self._load_vars


class RMSpropOptimizer(object):
    def __init__(self, model, grads, tvars, decay=0.9): #TODO:replace with 0.9
        self._decay = decay
        self._config = model.config
        self._eps = model.config.opt_eps
        self._max_update_norm = model.config.max_update_norm
        self._lr = model.config.lr

        self._grads = grads
        self._tvars = tvars

        self._ms = []
        self._ms_accu_op = []
        for tvar, g in zip(tvars, grads):
            self._ms.append(tf.get_variable(g.op.name + "_ms",
                                                     initializer=tf.ones_like(tvar, dtype=tf.float32) / 100,
                                                     trainable=False))

            g = tf.convert_to_tensor(g)
            with tf.name_scope("set_global_vars"), tf.control_dependencies([g]):
                self._ms_accu_op.append(tf.assign(self._ms[-1],
                                                  self._decay * self._ms[-1] + (1 - self._decay) * tf.square(g)))

        self._updates = list()
        self._train_op = list()

        if self._config.opt_inverse_type == "add":
            logger.info("inversion stability by adding epsilon")
        elif self._config.opt_inverse_type == "pseudo":
            logger.info("inversion stability by thresholding eigen-values by epsilon")

        # compute updates
        with tf.control_dependencies(self._ms_accu_op):
            for grad, ms in zip(grads, self._ms):
                self._updates.append(tf.multiply(self._lr,self.update(grad, ms)))

        # clip updates
        if self._max_update_norm > 0:
            logger.info("clipping total update")
            self._updates, _ = tf.clip_by_global_norm(self._updates, self._max_update_norm)

        # apply updates op

        for i, tvar in enumerate(tvars):
            self._train_op.append(tf.assign_add(tvar, -self._updates[i]))


    def update(self, grad, ms):
        if self._config.opt_inverse_type == "add":
            update = grad / (ms + self._eps)
        elif self._config.opt_inverse_type == "pseudo":
            condition = tf.greater_equal(ms, self._eps)
            update = tf.where(condition, grad / ms, grad)
        else:
            raise ValueError("opt_inverse_type has invalid value")

        return update

    @property
    def updates(self):
        return self._updates

    @property
    def train_op(self):
        return self._train_op

    @property
    def grads(self):
        return self._grads

    def get_grads_norm(self):
        g_norm = []
        for grad in self._grads:
            g_norm.append(tf.reduce_sum(tf.square(grad)))

        ms_norm = []
        for ms in self._ms:
            ms_norm.append(tf.reduce_min(ms))

        u_norm = []
        for update in self._updates:
            u_norm.append(tf.reduce_sum(tf.square(update)))
        return tf.sqrt(tf.add_n(g_norm)), tf.reduce_min(tf.stack(ms_norm)), tf.sqrt(tf.add_n(u_norm))

    def clip_by_layer(self):
        L = self._config.lstm_layers_num
        updates = list()
        clipped, _ = tf.clip_by_global_norm([self._updates[0],self._updates[1]], self._max_update_norm)
        updates.extend(clipped)
        for i in range(L):
            k = 2 + 2 * i
            clipped, _ = tf.clip_by_global_norm([self._updates[k], self._updates[k + 1]],
                                                  self._max_update_norm)
            updates.extend(clipped)

        updates.append(tf.clip_by_norm(self._updates[-1], self._max_update_norm))

        return updates

    def get_ms_max(self):
        ms_max = []
        for ms in self._ms:
            ms_max.append(tf.reduce_max(ms))

        return tf.reduce_max(tf.stack(ms_max))

    def get_ms(self):
        ms = []
        for m in self._ms:
            m = tf.convert_to_tensor(m)
            ms.append(tf.reshape(m,shape=[-1]))

        return tf.concat(ms, 0)

    def get_grad(self):
        gs = []
        for g in self._grads:
            g = tf.convert_to_tensor(g)
            gs.append(tf.reshape(g,shape=[-1]))

        return tf.concat(gs, 0)


def print_tvars():
    tvars = tf.trainable_variables()
    # print(tvars)
    nvars = config.embedding_size * config.vocab_size
    for var in tvars[1:]:
        sh = var.get_shape().as_list()
        nvars += np.prod(sh)
        # print ('var: %s, size: [%s]' % (var.name,', '.join(map(str, sh))))
    logger.info('%2.2fM variables'% (nvars*1e-6))


def tvars_num():
    tvars = tf.trainable_variables()
    nvars = config.embedding_size * config.vocab_size
    for var in tvars[1:]:
        sh = var.get_shape().as_list()
        nvars += np.prod(sh)
    return nvars


def run_epoch(session, model, eval_op=None, verbose=True):
    """run the given model over its data"""
    if eval_op is not None:
        model.input.shuffle()

    start_time = time.time()
    losses = 0.0
    iters = 0

    # zeros initial state for all devices
    state = session.run(model.initial_state)

    feed_dict_masks = {}
    # if variational every epoch --> update masks
    if config.variational == 'epoch' and eval_op is not None:
        # generate masks for epoch
        feed_dict_masks = model.gen_masks(session)
        feed_dict_masks.update(model.gen_emb_mask(session))
        # randomize words to drop from the vocabulary
        if config.drop_i < 1.0:
            words2drop = list()
            for i in range(config.batch_size):
                rand = np.random.rand(config.vocab_size)
                bin_vec = np.zeros(config.vocab_size, dtype=np.int32)
                bin_vec[rand > config.drop_i] = 1
                drop = np.where(bin_vec == 1)
                words2drop.append(drop[0])
            dropped = list()

    # evaluate loss and final state for all devices
    fetches = {
        "loss": model.loss,
        "final_state": model.final_state
    }

    # perform train op if training
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        # if variational every batch --> update masks
        if config.variational == 'batch' and eval_op is not None:
            feed_dict_masks = model.gen_masks(session)

        # pass states between time batches
        feed_dict = dict(feed_dict_masks.items())
        for j, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[j].c
            feed_dict[h] = state[j].h

        feed_dict.update(model.input.get_batch(step*model.input.time_steps))

        if eval_op is not None:
            if not config.drop_embed_var:
                feed_dict.update(model.gen_emb_mask(session))

            if config.drop_i < 1.0:
                for i, batch in enumerate(feed_dict[model.input.input_data]):
                    for j, w in enumerate(batch):
                        if w in words2drop[i]:
                            dropped.append(w)
                            feed_dict[model.emb_mask][i, j, :] = 0

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]
        state = vals["final_state"]

        losses += loss
        iters += model.input.time_steps


        if verbose and step % (model.input.epoch_size // 10) == 10:
            logger.info("%.3f perplexity: %.3f bits: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(losses / iters), np.log2(np.exp(losses / iters)),
                   iters * model.input.batch_size / (time.time() - start_time)))

    if eval_op is not None and config.drop_i < 1.0:
        logger.info("dropped %d/%d words" % (len(dropped), model.input.data_len))

    return np.exp(losses / iters)


def train_optimizer(session, layer, m, mvalid, train_writer, valid_writer, saver):
    """ Trains the network by the given optimizer """

    global bestVal

    def stop_criteria(epoch):
        stop_window = 5
        if epoch == epochs_num - 1:
            return True

        if len(valid_perplexity) > stop_window:
            diff = [valid_perplexity[k] - valid_perplexity[k+1] for k in range(-stop_window, -1,  1)]
            if np.max(diff) < 0:
                return True
            else:
                return False
        else:
            return False

    epochs_num = config.layer_epoch

    logger.info("updating dropout probabilities")
    m.update_drop_params(session, config.drop_output, config.drop_state)

    if args.save:
        logger.info("saving initial model....\n")
        save_path = saver.save(session, directory + '/saver/best_model' + str(layer))
        logger.info("save path is: %s" % save_path)

    lr_decay = config.lr_decay
    current_lr = session.run(m.lr)


    if config.opt == "asgd" or config.opt == "arms":
        validation_tolerance = 5

    if config.opt == "masgd" or config.opt == "marms":
        lr_decay = 1.0

    valid_perplexity = []
    i = session.run(m.epoch)
    should_stop = False
    while not should_stop:
        start_time = time.time()
        if len(valid_perplexity) >= 2 and valid_perplexity[-1] > valid_perplexity[-2]:
            current_lr *= lr_decay
            m.assign_lr(session, current_lr)
            lr_sum = tf.Summary(value=[tf.Summary.Value(tag="learning_rate_track" + str(layer),
                                                       simple_value=current_lr)])
            train_writer.add_summary(lr_sum, i + 1)
            if config.opt == "asgd" or config.opt == "arms" and not session.run(m.optimizer.trigger):
                validation_tolerance -= 1
                if validation_tolerance == 0:
                    logger.info("setting trigger and T")
                    m.optimizer.set_trigger(session)
                    m.optimizer.set_T(session, i)
                    lr_decay = 1.0

        logger.info("Epoch: %d Learning rate: %.3f Max Update Norm: %.3f" % (i + 1, session.run(m.lr), config.max_update_norm))

        if config.opt == "asgd" or config.opt == "arms":
            logger.info("Trigger is %s" % bool(session.run(m.optimizer.trigger)))

        ###################################### train ######################################
        train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        train_sum = tf.Summary(value=[tf.Summary.Value(tag="train_perplexity_layer" + str(layer),
                                                       simple_value=train_perplexity)])
        train_writer.add_summary(train_sum, i + 1)
        logger.info("Epoch: %d Train Perplexity: %.3f Bits: %.3f " % (i + 1, train_perplexity, np.log2(train_perplexity)))

        if ((config.opt == "asgd" or config.opt == "arms") and session.run(m.optimizer.trigger)) \
                or config.opt == "marms" or config.opt == "masgd":
            logger.info("saving model weights....")
            session.run(m.optimizer.save_vars)
            logger.info("setting averaged weights....")
            session.run(m.optimizer.final_assign_op)

        ###################################### valid ######################################
        valid_perplexity.append(run_epoch(session, mvalid, verbose=False))
        valid_sum = tf.Summary(value=[tf.Summary.Value(tag="valid_perplexity_layer" + str(layer),
                                                       simple_value=valid_perplexity[-1])])
        valid_writer.add_summary(valid_sum, i + 1)
        logger.info("Epoch: %d Valid Perplexity: %.3f Bits: %.3f" % (i + 1, valid_perplexity[-1], np.log2(valid_perplexity[-1])))

        # save model only when validation improves
        if bestVal > valid_perplexity[-1]:
            bestVal = valid_perplexity[-1]
            if args.save:
                try:
                    save_path = saver.save(session, directory + '/saver/best_model' + str(layer))
                    # print("save path is: %s" % save_path)
                    logger.info("save path is: %s" % save_path)
                except:
                    pass

        if config.opt == "asgd" or config.opt == "arms" and not session.run(m.optimizer.trigger):
            should_stop = False if i != epochs_num - 1 else True
        else:
            should_stop = stop_criteria(i)

        if ((config.opt == "asgd" or config.opt == "arms") and session.run(m.optimizer.trigger)) \
                or config.opt == "marms" or config.opt == "masgd":
                logger.info("loading model weights....")
                session.run(m.optimizer.load_vars)


        elapsed = time.time() - start_time
        logger.info("Epoch: %d took %02d:%02d\n" % (i + 1, elapsed // 60, elapsed % 60))
        i = m.epoch_inc(session)

    logger.info("restoring best model of current optimizer....")
    saver.restore(session, directory + '/saver/best_model' + str(layer))
    m.epoch_reset(session)


def test(session, m, mvalid, mtest):
    """ Trains the network by the given optimizer """

    start_time = time.time()
    logger.info("train:")
    train_perplexity = run_epoch(session, m)
    logger.info("valid:")
    valid_perplexity = run_epoch(session, mvalid)
    logger.info("test:")
    test_perplexity = run_epoch(session, mtest)
    logger.info("Train Perplexity: %.3f Bits: %.3f " % (train_perplexity, np.log2(train_perplexity)))
    logger.info("Valid Perplexity: %.3f Bits: %.3f " % (valid_perplexity, np.log2(valid_perplexity)))
    logger.info("Test Perplexity: %.3f Bits: %.3f " % (test_perplexity, np.log2(test_perplexity)))

    elapsed = time.time() - start_time
    logger.info("Evaluation took %02d:%02d" % (elapsed // 60, elapsed % 60))

    return  train_perplexity, valid_perplexity, test_perplexity


def read_flags(config, args):
    # assign flags into config
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            if key == "drop_state" or key == "drop_output":
                val = ast.literal_eval(val)
            if key == "layer_epoch":
                setattr(config, "entire_network_epoch", val)

            setattr(config, key, val)

    return config


def get_config(config_name):
    if config_name == "small":
        return config_pool.SmallConfig()
    if config_name == "big":
        return config_pool.BigConfig()
    if config_name == "bigger":
        return config_pool.BiggerConfig()
    if config_name == "test":
        return config_pool.TestConfig()
    else:
        raise ValueError("Invalid model: %s", config_name)


def print_config(config):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    logger.info('\n' + '\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))


def write_config(config, path):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    str = '\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs)
    f = open(path, "w")
    f.write(str)
    f.close()


def get_simulation_name(config):
    name = []
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None and key not in ["cpu_device", "gpu_devices", "start_layer", "ckpt_file"]:
            name.append(key + "-" + str(val).replace(",","-").replace(" ", "").replace("[", "").replace("]", ""))
    return "_".join(name)


def get_gpu_devices(str):
    devices_num = re.findall("[0-9]", str)
    return devices_num


def get_vars2restore(layer):
    if layer == 0:
        return None
    else:
        vars2load = []
        for var in tf.trainable_variables():
            lstm_idx = re.findall("lstm([0-9])+", var.op.name)
            if len(lstm_idx) == 0:
                vars2load.append(var)
            elif int(lstm_idx[0]) <= layer:
                vars2load.append(var)

        return vars2load


def write_to_summary(sum_path, config, train, valid, test):
    attr = sorted([attr for attr in dir(config) if not attr.startswith('__')])
    if not os.path.exists(sum_path):
        f = open("./summary.csv", "w")
        header = list()
        for arg in attr:
            header.append(arg)

        header.extend(["train_perp_tot", "valid_perp_tot", "test_perp_tot", "train_perp_0", "valid_perp_0", "test_perp_0"])

        f.write(",".join(header) +"\n")
    else:
        f = open("./summary.csv", "a")

    sum_list = list()
    for arg in attr:
        sum_list.append(str(getattr(config, arg)).replace(",","-").replace(" ", ""))

    scores = [[str(t), str(v), str(ts)] for t, v, ts in zip(train, valid, test)]
    sum_list.extend(scores[-1])
    scores.pop()
    scores = [str(item) for sublist in scores for item in sublist]
    sum_list.extend(scores)
    f.write(",".join(sum_list) +"\n")
    f.close()


def remove_tempstate_files(dir):
    for (folder, subs, files) in os.walk(dir):
        for filename in files:
            tempstate_files = [f for f in files if "tempstate" in f]
            if "tempstate" in filename:
                file_path = os.path.join(dir, filename)
                yield(file_path)


def main():

    ###################################### GL configs and restore ######################################
    GL = config.GL
    start_layer = 0 if config.GL else config.lstm_layers_num - 1

    if args.start_layer is not None:
        if args.ckpt_file is None:
            raise ValueError("must provide ckpt_file flag with start_layer_flag")
        print("\nstarting training from layer: %d\n" % args.start_layer)
        start_layer = args.start_layer - 1

    start_time_total = time.time()

    ###################################### GL training ######################################
    for layer in range(start_layer, config.lstm_layers_num):
        config.lstm_layers_num = layer + 1

        ###################################### build graph ######################################
        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
            tf.set_random_seed(seed)

            with tf.name_scope("Train"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    train_input = PTBInput(config=config, data=train_data)
                    m = PTBModel(is_training=True, config=config, inputs=train_input)
                train_writer = tf.summary.FileWriter(directory + '/train',
                                                     graph=tf.get_default_graph())
                logger.info("train shape: (%d,%d), train len: %d, epoch size: %d" %
                            (train_input.data.shape[0], train_input.data.shape[1], train_input.data_len, train_input.epoch_size))

            with tf.name_scope("Valid"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    valid_input = PTBInput(config=config, data=valid_data)
                    mvalid = PTBModel(is_training=False, config=config, inputs=valid_input)
                valid_writer = tf.summary.FileWriter(directory + '/valid')
                logger.info("valid shape: (%d,%d), valid len: %d, epoch size: %d" %
                            (valid_input.data.shape[0], valid_input.data.shape[1], valid_input.data_len,
                             valid_input.epoch_size))

            saver = tf.train.Saver(var_list=tf.trainable_variables())
            vars2load = get_vars2restore(layer)
            if vars2load is not None and GL:
                restore_saver = tf.train.Saver(var_list=vars2load)

            config.tvars_num = '%2.2fM' %(tvars_num()*1e-6)
            print_tvars()

        sess_config = tf.ConfigProto(device_count={"CPU": 2},
                                     inter_op_parallelism_threads=2,
                                     intra_op_parallelism_threads=8)

        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.visible_device_list = ",".join(get_gpu_devices(args.gpu_devices))

        ###################################### train ######################################
        with tf.Session(graph=graph, config=sess_config) as session:
            session.run(tf.global_variables_initializer())

            if vars2load is not None and GL:
                restore_saver.restore(session, directory + '/saver/best_model' + str(layer - 1))

            logger.info("training layer #%d" % (layer + 1))

            start_time = time.time()
            train_optimizer(session, layer, m, mvalid, train_writer, valid_writer, saver)
            elapsed = time.time() - start_time

            logger.info("optimization of layer %d took %02d:%02d:%02d\n" %
                  (layer + 1, elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

        tf.reset_default_graph()

        train_writer.close()
        valid_writer.close()

    elapsed = time.time() - start_time_total
    logger.info("optimization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))




    ###################################### GL evaluation ######################################
    if args.no_eval is None:
        batch_size = config.batch_size
        config.batch_size = 1
        train_perplexity, valid_perplexity, test_perplexity = [], [], []
        for layer in range(start_layer, config.lstm_layers_num):
            config.lstm_layers_num = layer + 1
            logger.info("Evaluating layer {0}".format(layer+1))

            ###################################### build graph ######################################
            with tf.Graph().as_default() as graph:
                tf.set_random_seed(seed)
                np.random.seed(seed)
                initializer = tf.random_uniform_initializer(-config.init_scale,
                                                            config.init_scale, seed=seed)

                with tf.name_scope("Train"):
                    train_input = PTBInput(config=config, data=train_data)
                    with tf.variable_scope("Model", reuse=None, initializer=initializer):
                        m = PTBModel(is_training=False, config=config, inputs=train_input)
                    logger.info("train shape: (%d,%d), train len: %d, epoch size: %d" %
                                (train_input.data.shape[0], train_input.data.shape[1], train_input.data_len, train_input.epoch_size))

                with tf.name_scope("Valid"):
                    valid_input = PTBInput(config=config, data=valid_data)
                    with tf.variable_scope("Model", reuse=True, initializer=initializer):
                        mvalid = PTBModel(is_training=False, config=config, inputs=valid_input)
                    logger.info("valid shape: (%d,%d), valid len: %d, epoch size: %d" %
                                (valid_input.data.shape[0], valid_input.data.shape[1], valid_input.data_len, valid_input.epoch_size))

                with tf.name_scope("Test"):
                    test_input = PTBInput(config=config, data=test_data)
                    with tf.variable_scope("Model", reuse=True, initializer=initializer):
                        mtest = PTBModel(is_training=False, config=config, inputs=test_input)
                    logger.info("test shape: (%d,%d), test len: %d, epoch size: %d" %
                                (test_input.data.shape[0], test_input.data.shape[1], test_input.data_len, test_input.epoch_size))
                saver = tf.train.Saver(var_list=tf.trainable_variables())

            sess_config = tf.ConfigProto(device_count={"CPU": 2},
                                         inter_op_parallelism_threads=2,
                                         intra_op_parallelism_threads=8)

            sess_config.gpu_options.allow_growth = True
            sess_config.gpu_options.visible_device_list = ",".join(get_gpu_devices(args.gpu_devices))

            ###################################### evaluation ######################################
            with tf.Session(graph=graph, config=sess_config) as session:
                session.run(tf.global_variables_initializer())
                saver.restore(session, directory + '/saver/best_model' + str(config.lstm_layers_num - 1))
                train_pp, valid_pp, test_pp = test(session, m, mvalid, mtest)
                train_perplexity.append(train_pp)
                valid_perplexity.append(valid_pp)
                test_perplexity.append(test_pp)

        config.batch_size = batch_size
        write_to_summary("./summary.csv", config, train_perplexity, valid_perplexity, test_perplexity)

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    for file_path in remove_tempstate_files(directory + '/saver'):
        os.remove(file_path)


if __name__ == "__main__":
    ###################################### argument parsing ######################################
    ap = argparse.ArgumentParser()

    ap.add_argument("--gpu_devices",        type=str, default="0", help="gpu device list")
    ap.add_argument("--cpu_device",         type=str, default="/cpu:0", help="cpu device")
    ap.add_argument("--ckpt_file",          type=str, default=None, help="file path for restore")
    ap.add_argument("--start_layer",        type=int, default=None, help="train from layer")
    ap.add_argument("--name",               type=str, default="debug", help="simulation name")
    ap.add_argument("--model",              type=str, default="small", help="model name")
    ap.add_argument("--seed",               type=int, default=None, help="seed")
    ap.add_argument("--data",               type=str, default="ptb", help="data type")
    ap.add_argument("--opt",                type=str, default=None, help="optimizer name")
    ap.add_argument("--opt_eps",            type=float, default=None, help="optimizer epsilon")
    ap.add_argument("--opt_inverse_type",   type=str, default=None, help="optimizer inversion type")
    ap.add_argument("--opt_clip_by_var",    dest='opt_clip_by_var', action='store_true', help="optimizer clip update by vars or globally")
    ap.add_argument("--opt_mom",            type=float, default=None, help="optimizer momentum")
    ap.add_argument("--opt_mom_decay",      type=float, default=None, help="optimizer momentum decay")
    ap.add_argument("--lr",                 type=float, default=None, help="training learning rate")
    ap.add_argument("--lr_decay",           type=float, default=None, help="learning rate decay")
    ap.add_argument("--max_update_norm",    type=float, default=None, help="max update norm")
    ap.add_argument("--batch_size",         type=int, default=None, help="batch size")
    ap.add_argument("--time_steps",         type=int, default=None, help="bptt truncation")
    ap.add_argument("--units_num",          type=int, default=None, help="#of units in lstm layer")
    ap.add_argument("--embedding_size",     type=int, default=None, help="#of units in embedding")
    ap.add_argument("--layer_epoch",        type=int, default=None, help="epochs per layer")
    ap.add_argument("--lstm_layers_num",    type=int, default=None, help="#of lstm layers")
    ap.add_argument("--AR",                 type=float, default=None, help="activation regularization parameter")
    ap.add_argument("--TAR",                type=float, default=None, help="temporal activation regularization parameter")
    ap.add_argument("--drop_output",        type=str, default=None, help="list of dropout parameters for outer connections")
    ap.add_argument("--drop_state",         type=str, default=None, help="list of dropout parameters for recurrent connections")
    ap.add_argument("--keep_prob_embed",    type=float, default=None, help="keep prob for embedding")
    ap.add_argument("--opt_c_lipsc",        type=float, default=None, help="for lwgc")
    ap.add_argument("--drop_i",             type=float, default=None, help="drop words")
    ap.add_argument("--no_eval",            dest='no_eval', action='store_true')
    ap.add_argument("--GL",                 dest='GL', action='store_false')
    ap.add_argument('--verbose',            dest='verbose', action='store_true')
    ap.add_argument('--save',               dest='save', action='store_true')
    ap.add_argument('--drop_embed_var',     dest='drop_embed_var', action='store_true')

    ap.set_defaults(no_eval=None)
    ap.set_defaults(drop_embed_var=None)
    ap.set_defaults(opt_clip_by_var=None)
    ap.set_defaults(GL=None)
    ap.set_defaults(verbose=None)
    ap.set_defaults(save=True)
    args = ap.parse_args()

    ###################################### data read & general configs ######################################
    config_pool = ptb_config
    reader = ptb_reader.ptb_raw_data

    config = get_config(args.model)
    config = read_flags(config, args)

    if args.data == 'ptb':
        config.vocab_size = 10000
    elif args.data == 'wiki2':
        config.vocab_size = 33278
    else:
        raise ValueError("invalid data-set name")

    if args.seed is not None:
        config.seed = seed = args.seed
    elif config.seed == 0:
        config.seed = seed = np.random.randint(0,1000000)
    else:
        seed = config.seed

    simulation_name = get_simulation_name(config)
    model_config_name = args.model

    directory = "./results/" + simulation_name + "_seed" + str(seed)
    data_path = "./data"

    if args.save:
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '/train')
            os.makedirs(directory + '/test')
            os.makedirs(directory + '/saver')
        elif args.ckpt_file is None:
            raise ValueError("simulation already exists; rerun with name flag")


    logFormatter = logging.Formatter("%(asctime)-15s | %(levelname)-8s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("{0}/logger.log".format(directory))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if args.verbose:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)



    logger.info("results are saved into: {dir}".format(dir=directory))


    raw_data = reader(data_path, args.data)
    train_data, valid_data, test_data, _ = raw_data

    logger.info("cmd line: python " + " ".join(sys.argv) )

    logger.info("Simulation configurations" )
    print_config(config)

    bestVal = config.vocab_size

    try:
        sys.exit(main())
    except Exception:
        logger.exception(Exception)
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        if args.save:
            if os.path.exists(directory):
                shutil.rmtree(directory)


# TODO: add norms method for every optimizer
