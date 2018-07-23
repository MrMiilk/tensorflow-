import tensorflow as tf
from .bins import *


def conv_layer(input, parameters):
    pass


def pooling_layer(input, parameters):
    pass


def conv_pool_block(input, parameters):
    a = input
    num_conv, conv_params, pool_params = parse_param("c_p_block", parameters)
    for l in range(num_conv):
        a = conv_layer(input=a, parameters=conv_params)
    output = pooling_layer(input=a, parameters=pool_params)
    return output


def FC_layer(input, parameters):
    pass