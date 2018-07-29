from vgg16 import *
from config import *


def loss(label, loggit):
    with tf.name_scope("loss_func"):
        return label * tf.log(loggit)

if __name__ == '__main__':
    layer_types = layer_types
    layer_params = layer_params
    learning_rate = 2e-3
    output_shape = [None, 20]
    model = VGG_16(layer_types, layer_params, output_shape)
    model.train_setting(loss=loss, optimzer="adam", learning_rate=learning_rate)
    model.train()