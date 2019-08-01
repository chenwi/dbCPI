# coding:utf-8

import tensorflow as tf


def feedforward(inputs,
                num_filters=None,
                scope="multihead_attention",
                reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, kernel_size=1, activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.constant_initializer(0.01)
        )

        # outputs = tf.tanh(outputs) * tf.sigmoid(outputs)
        # tf.nn.softplus

        outputs = tf.layers.conv1d(
            inputs=outputs, filters=num_filters, kernel_size=1, activation=None, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.constant_initializer(0.01)
        )
        # outputs = tf.tanh(outputs) * tf.sigmoid(outputs)
        outputs += inputs  # Residual connection
        outputs = normalize(outputs)

    return outputs


def res_block(inputs, hidden_size):
    outputs = feedforward(inputs=inputs, num_filters=hidden_size)
    # outputs = feedforward(inputs=outputs, num_filters=hidden_size)
    # outputs = feedforward(inputs=outputs, num_filters=hidden_size)
    # outputs = feedforward(inputs=outputs, num_filters=hidden_size)
    return outputs


def conv2d(inputs, filters, kernel_size, scope=None):
    with tf.name_scope(scope):
        return tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                activation=tf.nn.relu, use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.constant_initializer(0.01)
                                )


def gate_cnn(inputs, filters, kernel_size):
    outputs = tf.layers.conv2d(inputs=inputs, filters=filters,
                               kernel_size=kernel_size,
                               activation=None, use_bias=True,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                               bias_initializer=tf.constant_initializer(0.01)
                               )
    outputs1 = tf.layers.conv2d(inputs=inputs, filters=filters,
                                kernel_size=kernel_size,
                                activation=None, use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.constant_initializer(0.0)
                                )

    outputs = tf.tanh(outputs) * tf.sigmoid(outputs1)
    # outputs = tf.nn.relu(outputs)
    return outputs


def max_pool(inputs, length, num_filters):
    outputs = tf.reshape(inputs, [-1, length, num_filters])
    outputs = tf.nn.relu(outputs)
    pooled = tf.layers.max_pooling1d(outputs, length, 1)
    pooled_flatten = tf.reshape(pooled, [-1, num_filters])
    return pooled_flatten


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=tf.AUTO_REUSE):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def bilstm(inputs, hidden_size, scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        def _rnn_cell(num_units, name):
            return tf.nn.rnn_cell.GRUCell(
                num_units,
                # initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                name=name
            )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            _rnn_cell(hidden_size, name=f'{scope}_fw'),
            _rnn_cell(hidden_size, name=f'{scope}_bw'),
            inputs=inputs,
            # sequence_length=seq_length,
            dtype=tf.float32
        )
    return outputs
