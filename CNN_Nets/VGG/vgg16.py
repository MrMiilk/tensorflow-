import tensorflow as tf
from blocks import *


class VGG_16(object):

    def __init__(self, layer_types, layer_params, output_shape, input_size=224, log_dir="./logs"):
        self.num_layers = len(layer_params)
        #untrainable
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 3])
        self.Y = tf.placeholder(dtype=tf.float32, shape=output_shape)
        self.global_step = tf.Variable(0, trainable=False)
        self.layer_type = layer_types
        self.layer_params = layer_params
        self.output = None
        self.loss = None
        self.optimzer = None
        self.train_op = None
        self.log_dir = log_dir
        self.writer = None

        self._model()

    def _model(self):
        a = self.input
        for l in range(self.num_layers):
            l_type = self.layer_type[l]
            l_params = self.layer_params[l]
            print("building block:" + str(l))
            if l_type == "c_p":
                a = conv_pool_block(a, parameters=l_params, block_index=l)
            elif l_type == "fc":
                a = FC_layer(a, parameters=l_params, layer_index=l)
            elif l_type == 'ft':
                a = flatten_block(a, parameters=l_params)
            else:
                raise ValueError("unsupport block type: %s" % l_type)
            print("output tensor:", a)
        self.output = tf.nn.softmax(a)

    def train_setting(self, loss, optimzer, learning_rate):
        with tf.name_scope("train"):
            self.loss = loss(self.output, self.Y)

            if optimzer == 'adam':
                self.optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimzer == 'sgd':
                self.optimzer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            else:
                raise ValueError("unsupport optimzer: %s" % optimzer)
            with tf.name_scope("train_step"):
                self.train_op = self.optimzer.minimize(self.loss, global_step=self.global_step)

    def train(self):
        inizer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(inizer)
            self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)

    def predict(self):
        pass