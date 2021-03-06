import tensorflow as tf
import numpy as np


def parse_param(flag, parameters):
    if flag == "conv":
        return parameters[0:2], parameters[2:]
    if flag == "af":
        return parameters[0:-1], parameters[-1]
    if flag == "pooling":
        return parameters
    if flag == "c_p_block":
        num_conv = len(parameters[0])
        return num_conv, parameters[0], parameters[1]
    if flag == "flatten":
        return parameters[0]
    else:
        raise ValueError("no such flag: %s" % flag)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))