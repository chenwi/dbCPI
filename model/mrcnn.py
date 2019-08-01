from model.attention import *
from model.basemodel import *

import tensorflow as tf

# multi-channels ResCNN
class MRCNN():
    def __init__(
            self, sequence_length, num_classes,
            word_embedding_size,
            position_size, position_embedding_size,
            filter_size, num_filters,
            l2_reg_lambda,
            gamma, m_plus, m_minus,
            artificial_class_index,

    ):
        self.input_x1 = tf.placeholder(tf.int64, [None, sequence_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int64, [None, sequence_length], name='input_x2')
        self.input_p1 = tf.placeholder(tf.int64, [None, sequence_length], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int64, [None, sequence_length], name='input_p2')
        self.input_y_plus = tf.placeholder(tf.int64, [None, 2], name='input_y_plus')
        self.input_c_minus = tf.placeholder(tf.int64, [None, num_classes - 1, 2], name='input_c_minus')
        self.Embedding_W1 = tf.placeholder(tf.float32, [None, word_embedding_size], name='Embedding_W1')
        self.Embedding_W2 = tf.placeholder(tf.float32, [None, word_embedding_size], name='Embedding_W2')
        self.input_y = tf.placeholder(tf.int64, name='input_y')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.hidden_size = word_embedding_size + 2 * position_embedding_size

        self.embedding_x1 = tf.nn.embedding_lookup(self.Embedding_W1, self.input_x1)
        self.embedding_x2 = tf.nn.embedding_lookup(self.Embedding_W2, self.input_x2)

        with tf.name_scope('position_embedding'):
            Embedding_zero = tf.Variable(
                tf.zeros([1, position_embedding_size]),
                name='embedding_zero',
                dtype=tf.float32,
                trainable=False
            )
            Embedding_normal = tf.get_variable(
                'embedding_normal',
                shape=[position_size - 1, position_embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.Embedding_P = tf.concat([Embedding_zero, Embedding_normal], axis=0)

            self.embedding_p1 = tf.nn.embedding_lookup(self.Embedding_P, self.input_p1)
            self.embedding_p2 = tf.nn.embedding_lookup(self.Embedding_P, self.input_p2)

        with tf.name_scope('embedding_inputs'):
            self.inputs1 = tf.concat([self.embedding_x1, self.embedding_p1, self.embedding_p2], 2)
            self.inputs2 = tf.concat([self.embedding_x2, self.embedding_p1, self.embedding_p2], 2)

        with tf.name_scope('resunit'):
            inputs_res1 = res_block(inputs=self.inputs1, hidden_size=self.hidden_size)
            inputs_res2 = res_block(inputs=self.inputs2, hidden_size=self.hidden_size)

        with tf.name_scope('attention1'):
            # o_fw, o_bw = bilstm(inputs=self.inputs1, hidden_size=self.hidden_size // 2, scope='bilstm11')
            # o_fw1, o_bw1 = bilstm(inputs=o_fw + o_bw, hidden_size=self.hidden_size // 2, scope='bilstm12')
            # inputs_rnn1 = tf.concat([o_fw + o_bw, o_fw1 + o_bw1], axis=-1)

            o_fw, o_bw = bilstm(inputs=self.inputs1, hidden_size=self.hidden_size, scope='bilstm13')
            inputs_att1 = attention(o_fw + o_bw, self.hidden_size, sequence_length, scope='att1',
                                    num_heads=3)
            # inputs_att1 = res_block(inputs_att1, hidden_size=self.hidden_size)

        with tf.name_scope('attention2'):
            # o_fw, o_bw = bilstm(inputs=self.inputs2, hidden_size=self.hidden_size // 2, scope='bilstm21')
            # o_fw1, o_bw1 = bilstm(inputs=o_fw + o_bw, hidden_size=self.hidden_size // 2, scope='bilstm22')
            # inputs_rnn2 = tf.concat([o_fw + o_bw, o_fw1 + o_bw1], axis=-1)
        #
            o_fw, o_bw = bilstm(inputs=self.inputs2, hidden_size=self.hidden_size, scope='bilstm23')
            inputs_att2 = attention(o_fw + o_bw, self.hidden_size, sequence_length, scope='att2',
                                    num_heads=3)
            inputs_att2 = res_block(inputs_att2, hidden_size=self.hidden_size)

        with tf.name_scope('multicnn'):
            embedding_r1 = tf.expand_dims(self.inputs1, axis=-1)  # no res
            embedding_r2 = tf.expand_dims(self.inputs2, axis=-1)  # no res
            embedding_res1 = tf.expand_dims(inputs_res1, axis=-1)
            embedding_res2 = tf.expand_dims(inputs_res2, axis=-1)

            # embedding_rnn1 = tf.expand_dims(inputs_rnn1, axis=-1)  # no res
            # embedding_rnn2 = tf.expand_dims(inputs_rnn2, axis=-1)  # no res
            embedding_att1 = tf.expand_dims(inputs_att1, axis=-1)  # no res
            # embedding_att2 = tf.expand_dims(inputs_att2, axis=-1)  # no res

            embedding_all = tf.concat(
                [
                    embedding_r1,
                    embedding_r2,
                    embedding_res1,
                    embedding_res2,
                    # embedding_rnn1,
                    # embedding_rnn2,
                    embedding_att1,
                    # embedding_att2
                ], axis=-1)
            #
            outputs_cnn = gate_cnn(inputs=embedding_all,
                                   filters=num_filters,
                                   kernel_size=[filter_size, self.hidden_size])
            self.pooled_flatten = max_pool(outputs_cnn,
                                           length=sequence_length - filter_size + 1,
                                           num_filters=num_filters)
            # pooled_flatten = tf.layers.max_pooling1d(outputs, sequence_length, 1)
            # pooled_flatten = tf.reshape(pooled_flatten, [-1, self.hidden_size])
            #
            # self.pooled_flatten = tf.layers.max_pooling1d(outputs, sequence_length, 1)
            # self.pooled_flatten = tf.reshape(self.pooled_flatten, [-1, self.hidden_size])

        with tf.name_scope('score_prediction'):
            # self.pooled_flatten = tf.layers.dense(inputs=self.pooled_flatten,
            #                               units=100,
            #                               use_bias=True,
            #                               bias_initializer=tf.constant_initializer(0.01),
            #                               activation=tf.nn.relu)
            # pooled_flatten= tf.layers.dense(inputs=pooled_flatten,
            #                               units=100,
            #                               use_bias=True,
            #                               bias_initializer=tf.constant_initializer(0.01),
            #                               activation=tf.nn.relu)
            # self.pooled_flatten=tf.concat([self.pooled_flatten,pooled_flatten],axis=-1)
            self.scores = tf.layers.dense(inputs=self.pooled_flatten,
                                          units=num_classes,
                                          use_bias=True,
                                          bias_initializer=tf.constant_initializer(0.01),
                                          # kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda),
                                          activation=None)  # batch_size * num_classes
            # self.scores=lstm_scores

            self.max_score = tf.reduce_max(self.scores, axis=1)  # batch_size
            max_score_index = tf.argmax(self.scores, axis=1, name='predictions')  # batch_size
            predict_artificial_mask = tf.cast(tf.less(self.max_score, 0.), tf.int64)

            self.predictions = predict_artificial_mask * artificial_class_index + (
                    1 - predict_artificial_mask) * max_score_index

        with tf.name_scope('loss'):
            y_plus_score = tf.gather_nd(self.scores, self.input_y_plus)  # batch_size
            c_minus_gather = tf.gather_nd(self.scores, self.input_c_minus)  # batch_size * (num_classes - 1)
            c_minus_score = tf.reduce_max(c_minus_gather, axis=1)  # batch_size

            natual_losses = tf.log(1 + tf.exp(gamma * (m_plus - y_plus_score))) + tf.log(
                1 + tf.exp(gamma * (m_minus + c_minus_score)))
            artificial_losses = tf.log(1 + tf.exp(gamma * (m_minus + self.max_score)))

            ground_artificial_mask = tf.cast(tf.equal(self.input_y, artificial_class_index), tf.float32)
            losses = ground_artificial_mask * artificial_losses + (1.0 - ground_artificial_mask) * natual_losses

            l2_reg_loss = tf.constant(0.)
            for var in tf.trainable_variables():
                if var.name == self.embedding_x1.name or var.name == self.embedding_x2.name:
                    continue
                l2_reg_loss += tf.nn.l2_loss(var)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_reg_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        with tf.name_scope('optimize'):
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            # optimizer=tf.train.AdadeltaOptimizer(self.learning_rate)
            # grads_over_vars = optimizer.compute_gradients(self.loss)
            # self.train_op = optimizer.apply_gradients(grads_over_vars, global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        with tf.name_scope('summary'):
            loss_summary = tf.summary.scalar('loss', self.loss)
            acc_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            self.dev_summary_op = acc_summary
