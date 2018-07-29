import tensorflow as tf
from blocks import *
from input_data import *


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
        Input_Queue = Input_data('.\\DS2\\*.*')
        inizer = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(inizer)
            self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            batch_X, batch_y = Input_Queue.input_pipline(batch_size=30)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                for _ in range(1000):
                    if not coord.should_stop():
                        bx, by = sess.run([batch_X, batch_y])
                        sess.run(self.train_op, feed_dict={self.input:bx, self.Y:by})
                        print(_)
            except tf.errors.OutOfRangeError:
                print(('Catch out of range'))
            finally:
                coord.request_stop()
            coord.join(threads)

    def predict(self):
        pass