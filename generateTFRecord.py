# 将train和test写入到tfrecord的同时进行标注

import os
import cv2

import tensorflow as tf


train_path = './train'
test_path = './test'
classes = {i: i for i in range(1, 41)}
writer_train = tf.io.TFRecordWriter("orl_train.tfrecords")
writer_test = tf.io.TFRecordWriter("orl_test.tfrecords")

def generateTFRecoder():
    # 遍历字典
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for index, name in enumerate(classes):
        print(index)
        train = train_path + '/' + str(name) + '/'
        test = test_path + '/' + str(name) + '/'
        print(train)
        # 训练集
        # s.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        for img_name in os.listdir(train):
            # 每个图片地址
            img_path = train + img_name
            img = cv2.imread(img_path)
            # cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
            # cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
            # cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index + 1])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer_train.write(example.SerializeToString())
        # 测试集
        for img_name in os.listdir(test):
            img_path = test + img_name  # 每一个图片的地址
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index + 1])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer_test.write(example.SerializeToString())

    writer_test.close()
    writer_train.close()
generateTFRecoder()





