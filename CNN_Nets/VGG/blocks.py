import tensorflow as tf
from .bins import *


def conv_layer(input, parameters, block_index, layer_index):
    """
    解析参数，生成filter，添加指定卷积
    这里默认filter的initializer是tf.random_normal_initializer(),bias都初始化为0
    layer_name为conv_b_l形式
    :param input:
    :param parameters:
    :param block_index:
    :param layer_index:
    :return:
    """
    layer_name = "conv" + str(block_index) + "_" + str(layer_index)
    variable_params, layer_params = parse_param('conv', parameters)
    filter_shape, b_shape = variable_params
    nolinear_func, stride, padding = layer_params

    with tf.name_scope(layer_name):
        with tf.variable_scope(layer_name + "variables", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('Filter', filter_shape, dtype=tf.float32, initializer=tf.random_normal_initializer())
            b = tf.get_variable('bias', b_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        with tf.name_scope("conv"):
            output = tf.nn.conv2d(input, W, stride, padding) + b
        if hasattr(tf.nn, nolinear_func):
            with tf.name_scope(nolinear_func):
                nl_func = getattr(tf.nn, nolinear_func)
                output = nl_func(output)
    return output


def FC_layer(input, parameters, layer_index):
    variable_params, layer_params = parse_param('af', parameters)
    w_shape, b_shape = variable_params
    nolinear_func = layer_params
    layer_name = "FC" + str(layer_index)
    with tf.name_scope(layer_name):
        with tf.variable_scope(layer_name + "variables", reuse=tf.AUTO_REUSE):
            W = tf.get_variable("weight", tf.float32, w_shape, initializer=tf.random_normal_initializer())
            b = tf.get_variable("bias", tf.float32, b_shape, initializer=tf.random_normal_initializer())
        with tf.name_scope("FC"):
            output = tf.add(tf.matmul(input, W), b)
        if hasattr(tf.nn, nolinear_func):
            with tf.name_scope(nolinear_func):
                output = getattr(tf.nn, nolinear_func)(output)
        return output


def pooling_layer(input, parameters, block_index, layer_index):
    layer_params = parse_param('pooling', parameters)
    window_shape, pooling_type, padding = layer_params
    layer_name = "pooling" + str(block_index) + str(layer_index)
    with tf.name_scope(layer_name):
        output = tf.nn.pool(input, window_shape, pooling_type, padding)
    return output



def conv_pool_block(input, parameters, block_index):
    a = input
    num_conv, conv_params, pool_params = parse_param("c_p_block", parameters)
    for l in range(num_conv):
        a = conv_layer(input=a, parameters=conv_params, block_index=block_index, layer_index=l)
    output = pooling_layer(input=a, parameters=pool_params, block_index= block_index, layer_index=1)
    return output


def flatten_block(input, parameters):
    output_shape = parse_param("flatten", parameters)
    with tf.name_scope("flatten"):
        output = tf.reshape(input, [-1, output_shape])
    return output