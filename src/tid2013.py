"""
@File         : tid2013.py
@Time         : 2017/6/22 
@Author       : Chen Huang
@Update       : 
@Discription  : Builds the network.
"""


# from __future__ import unicode_literals
# from __future__ import print_function
# from __future__ import division

import numpy
import tensorflow as tf

import tid2013_input

# training parameters
INI_LEARNING_RATE = 0.0001
MAX_STEPS = int(3000 * (1800/4))
BATCH_SIZE = 4
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5000 # 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

EPSILON = 0.000001

DATA_TYPE = tf.float32

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def distorted_inputs(filenames):
    """Construct distorted input for training using the Reader ops.

    Args:
        :param filenames: list - [name1, name2, ...]

    Returns:
        :return patches_x, patches_y: 4D tensor of (batch_size*num_patches, patch_size, patch_size, depth). size.
        :return labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        :exception ValueError: If no data_dir
    """
    patches_x, patches_y, labels = tid2013_input.distorted_inputs(filenames=filenames, batch_size=BATCH_SIZE)

    return patches_x, patches_y, labels


def inputs(filenames):
    """Construct input without distortion for evaluation using the Reader ops.

    Args:
        :param filenames: list - [name1, name2, ...]

    Returns:
        :return patches_x, patches_y: 4D tensor of (batch_size*num_patches, patch_size, patch_size, depth). size.
        :return labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        :exception ValueError: If no data_dir
    """
    patches_x, patches_y, labels = tid2013_input.inputs(filenames=filenames, batch_size=BATCH_SIZE)

    return patches_x, patches_y, labels


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        :param x: Tensor
    """
    # # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # # session. This helps the clarity of presentation on tensorboard.
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # tf.summary.histogram(tensor_name + '/activations', x)
    # tf.summary.scalar(tensor_name + '/sparsity',
    #                                    tf.nn.zero_fraction(x))
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))


def conv_layer(input, kernel_shape, output_channels, activation):
    """ conv + relu
    
    Args:
        :param input: tensor. 
        :param kernel_shape: tuple - (height, width).
        :param output_channels: int.
        :param activation: func.
        
    Returns:
        :return output: tensor.
    """
    input_channel = input.get_shape()[3].value
    weights_initializer = tf.truncated_normal_initializer(
        stddev=1.0 / numpy.sqrt(float(kernel_shape[0]*kernel_shape[1]*input_channel)))
    weights = tf.get_variable(name='weights',
                              shape=kernel_shape + (input_channel, output_channels),
                              initializer=weights_initializer, dtype=tf.float32)
    biases = tf.get_variable(name='biases', shape=[output_channels],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    output = activation(conv + biases)

    return output


def extract_features(input):
    """ Extract features by alternative conv and max-pool.
    Args:
        :param input: tensor - (batch_size*num_patches_per_image, patch_size, patch_size, depth)
        
    Returns:
        :return pool5: tensor.
    """
    with tf.variable_scope('block1'):
        with tf.variable_scope('conv1_1'):
            conv1_1 = conv_layer(input, kernel_shape=(3, 3), output_channels=32, activation=tf.nn.relu)
            _activation_summary(conv1_1)

        with tf.variable_scope('conv1_2'):
            conv1_2 = conv_layer(conv1_1, kernel_shape=(3, 3), output_channels=32, activation=tf.nn.relu)
            _activation_summary(conv1_2)

        pool1 = tf.nn.max_pool(value=conv1_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool1')

    with tf.variable_scope('block2'):
        with tf.variable_scope('conv2_1'):
            conv2_1 = conv_layer(pool1, kernel_shape=(3, 3), output_channels=64, activation=tf.nn.relu)
            _activation_summary(conv2_1)

        with tf.variable_scope('conv2_2'):
            conv2_2 = conv_layer(conv2_1, kernel_shape=(3, 3), output_channels=64, activation=tf.nn.relu)
            _activation_summary(conv2_2)

        pool2 = tf.nn.max_pool(value=conv2_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool2')

    with tf.variable_scope('block3'):
        with tf.variable_scope('conv3_1'):
            conv3_1 = conv_layer(pool2, kernel_shape=(3, 3), output_channels=128, activation=tf.nn.relu)
            _activation_summary(conv3_1)

        with tf.variable_scope('conv3_2'):
            conv3_2 = conv_layer(conv3_1, kernel_shape=(3, 3), output_channels=128, activation=tf.nn.relu)
            _activation_summary(conv3_2)

        pool3 = tf.nn.max_pool(value=conv3_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool3')

    with tf.variable_scope('block4'):
        with tf.variable_scope('conv4_1'):
            conv4_1 = conv_layer(pool3, kernel_shape=(3, 3), output_channels=256, activation=tf.nn.relu)
            _activation_summary(conv4_1)

        with tf.variable_scope('conv4_2'):
            conv4_2 = conv_layer(conv4_1, kernel_shape=(3, 3), output_channels=256, activation=tf.nn.relu)
            _activation_summary(conv4_2)

        pool4 = tf.nn.max_pool(value=conv4_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool4')

    with tf.variable_scope('block5'):
        with tf.variable_scope('conv5_1'):
            conv5_1 = conv_layer(pool4, kernel_shape=(3, 3), output_channels=512, activation=tf.nn.relu)
            _activation_summary(conv5_1)

        with tf.variable_scope('conv5_2'):
            conv5_2 = conv_layer(conv5_1, kernel_shape=(3, 3), output_channels=512, activation=tf.nn.relu)
            _activation_summary(conv5_2)

        pool5 = tf.nn.max_pool(value=conv5_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool5')

    return pool5


def fully_connection(input, hiden_units, keep_prob):
    """ fully connection architecture.
    
    Args:
        :param input: tensor - (batch_size, ?)
        :param hiden_units: int.
        :param keep_prob: tensor. scalar.
    
    Returns:
        :return: tensor - (batch_size, 1)
    """
    input_units = input.get_shape()[1].value
    weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / numpy.sqrt(float(input_units)))
    with tf.variable_scope('block1'):
        weights = tf.get_variable(name='weights',
                                  shape=[input_units, hiden_units],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases = tf.get_variable(name='biases', shape=[hiden_units],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        hidden1 = tf.nn.relu(tf.matmul(input, weights) + biases)
        _activation_summary(hidden1)
        hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    with tf.variable_scope('block2'):
        weights = tf.get_variable(name='weights',
                                  shape=[hiden_units, 1],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases = tf.get_variable(name='biases', shape=[1],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        output = tf.matmul(hidden1_dropout, weights) + biases
        _activation_summary(output)

    return output


def inference(patches_x, patches_y, keep_prob):
    """Build the model up to where it may be used for inference.

    Args:
        :param patches_x, patches_y: tensor - (batch_size*num_patches_per_image, patch_size, patch_size, depth).
                Images placeholder, from inputs().
        :param keep_prob: tensor. scalar. used for fully_connection().

    Returns:
        :return output: tensor - (batch_size, )
    """
    with tf.variable_scope('extract_feature') as scope:
        feature_map_x = extract_features(patches_x)
        scope.reuse_variables()
        feature_map_y = extract_features(patches_y)

    with tf.variable_scope('fuse_features'):
        feature_x = tf.reshape(tensor=feature_map_x, shape=[tid2013_input.NUM_PATCHES_PER_IMAGE * BATCH_SIZE, -1])
        feature_y = tf.reshape(tensor=feature_map_y, shape=[tid2013_input.NUM_PATCHES_PER_IMAGE * BATCH_SIZE, -1])
        feature_dif = feature_x - feature_y

        features = tf.concat([feature_x, feature_y, feature_dif], axis=1)

    hiden_units = 512
    with tf.variable_scope('regression') as scope:
        patches_qualities = fully_connection(features, hiden_units, keep_prob)

    with tf.variable_scope('weighting') as scope:
        patches_weights = fully_connection(features, hiden_units, keep_prob)
        patches_weights = tf.nn.relu(patches_weights)
        _activation_summary(patches_weights)

        patches_weights = tf.reshape(patches_weights, (BATCH_SIZE, tid2013_input.NUM_PATCHES_PER_IMAGE)) + EPSILON
        patches_weights_sum = tf.expand_dims(tf.reduce_sum(patches_weights, axis=1), -1)
        patches_normalized_weights = patches_weights / patches_weights_sum
        _activation_summary(patches_normalized_weights)

    with tf.variable_scope('estimate_quality') as scope:
        patches_qualities = tf.reshape(patches_qualities, (BATCH_SIZE, tid2013_input.NUM_PATCHES_PER_IMAGE))

        output = patches_normalized_weights * patches_qualities
        output = tf.reduce_sum(output, axis=1)

    return output


def loss_func(scores, labels):
    """Mean absolute error.
    Args:
        :param scores: scores from inference().
        :param labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        :return Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.abs(scores - labels))
        tf.add_to_collection('losses', loss)

        return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        :param total_loss: Total loss from loss().

    Returns:
        :return loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        # tf.summary.scalar(l.op.name + '_average', loss_averages.average(l))

    return loss_averages_op


def train_func(data_num, total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = data_num / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INI_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        #opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
