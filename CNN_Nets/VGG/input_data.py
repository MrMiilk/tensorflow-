import tensorflow as tf
import os
import xml.dom.minidom
from bins import *
import numpy as np
import matplotlib.image as mpimg


class Input_data(object):
    """TODO:为训练提供数据供给"""
    def __init__(self, filenames_pattern, n_epochs=4, shuffle=True):
        self.filenames = tf.train.match_filenames_once(pattern=filenames_pattern)
        self.filenames_queue = tf.train.string_input_producer(string_tensor=self.filenames, num_epochs=n_epochs, shuffle=shuffle)
        self.reader = tf.TFRecordReader(name='Reader')
        self._read_node()

    def _read_node(self):
        _, value = self.reader.read(self.filenames_queue)
        features = tf.parse_single_example(
            value,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            }
        )
        self.images = tf.decode_raw(features['image'], tf.uint8)
        self.images = tf.reshape(self.images, [224, 224, 3])
        self.images = tf.cast(self.images, tf.float32)
        self.labels = tf.decode_raw(features['label'], tf.int64)
        self.labels = tf.reshape(self.labels, [20])
        self.labels = tf.cast(self.labels, tf.float32)

    def input_pipline(self, batch_size, min_after_dequeue=1024, ):
        capcity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch(
            [self.images, self.labels], batch_size, capcity, min_after_dequeue
        )
        return image_batch, label_batch


class Datas_hander(object):
    """TODO:预处理数据集"""
    def __init__(self, xml_path, img_path, TFR_path=None, file_size=500):
        self.xml_path = xml_path
        self.img_path = img_path
        self.TFR_path = TFR_path
        self.file_size = file_size
        self.class_map = dict(pottedplant=0, dog=1, horse=2, car=3, diningtable=4, boat=5, bus=6, bicycle=7,
                              motorbike=8, person=9, sofa=10, cow=11, aeroplane=12, sheep=13, cat=14, train=15,
                              chair=16, tvmonitor=17, bottle=18, bird=19)

    def resize_image(self, image, w, h):
        center_x = int(w/2)
        center_y = int(h/2)
        image = image[center_x-112:center_x+112, center_y-112:center_y+112]
        return image

    def init_label(self, obj):
        """
        :param obj:标签
        :return: 由于存在多类别的原因，label不是one_hot
        """
        num_target = len(obj)
        lables = []
        for o in range(num_target):
            tag = obj[o].getElementsByTagName("name")[0].firstChild.data
            lables.append(tag)
        lables = list(set(lables))
        label = np.zeros([1, 20], dtype=np.int64)
        for i in lables:
            label[0, self.class_map[i]] = 1
        return label.tostring()


    def open_image(self, image_ele, size):
        """
        :param image_ele:
        :return:
        """
        image_name = image_ele[0].firstChild.data
        image = mpimg.imread(self.img_path + "\\" + image_name)
        w = np.int64(size[0].getElementsByTagName('width')[0].firstChild.data)
        h = np.int64(size[0].getElementsByTagName('width')[0].firstChild.data)
        image = self.resize_image(image, w, h)
        return image

    def main(self):
        num_file = 1
        num = self.file_size
        writer = None
        for xml_bag in os.listdir(self.xml_path):
            #用于判断当前TFRecoder文件是否已满
            if num == self.file_size:
                if writer is not None:
                    writer.close()
                TFR_file_name = ".\\DS2\\data_%d.TFRecord" % num_file
                print('building: ', TFR_file_name)
                num_file += 1
                num = 0#当前TFRecoder容量
                writer = tf.python_io.TFRecordWriter(TFR_file_name)
            #获取，解析xml，写入TFRecoder
            dom = xml.dom.minidom.parse(self.xml_path + "\\" + xml_bag)
            root = dom.documentElement
            img_name = root.getElementsByTagName('filename')
            obj = root.getElementsByTagName('object')
            size = root.getElementsByTagName('size')
            label = self.init_label(obj)  # 获取label
            image = self.open_image(img_name, size)  # 获取图片，转码
            if image.shape[0] < 224 or image.shape[1] < 224:
                continue
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': bytes_feature(label),
                'image': bytes_feature(image.tostring()),
            }))
            writer.write(example.SerializeToString())
            num += 1
        writer.close()
        print("All file save to TFRrecode file")


if __name__ == '__main__':
    """调用此文件以执行数据预处理"""
    image_path = "D:\\notes\\Tensorflow_note\\CNN_Nets\\VGG\\DataSet\\JPEGImages"
    xml_path = "D:\\数据集\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\Annotations"
    hander = Datas_hander(xml_path, image_path)
    hander.main()
    # Temp = Input_data('.\\DS\\*.*')
    # with tf.Session() as sess:
    #     initer = tf.group(
    #         tf.global_variables_initializer(),
    #         tf.local_variables_initializer()
    #     )
    #     sess.run(initer)
    #     batch_X, batch_y = Temp.input_pipline(batch_size=10)
    #     print(batch_y)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for _ in range(1000):
    #         print(batch_X)
    #         print(sess.run(Temp.filenames))
    #         bx, by = sess.run([batch_X, batch_y])
    #         print(bx, by)
    #     coord.join(threads)