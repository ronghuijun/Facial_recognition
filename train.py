import tensorflow as tf
import numpy as np
import os
import cv2
import trainSet
import random
from sklearn.model_selection import train_test_split


SIZE = 28
# 配置CNN参数
REGULARIZATION_RATE = 0.0001    # 正则化参数
TRAINING_STEPS = 10000
MOVE_AVERAGE_DECAY = 0.99   # 滑动平均衰减率
LEARNING_RATE_BASE = 0.1    # 基础学习率
LEARNING_RATE_DECAY = 0.99     # 学习的衰减率
BATCH_SIZE = 20

MODEL_SAVE_PATH = './model/'
MODEL_NAME = "model.ckpt"

# 把所有label变成one_hot 放在一起
def get_label(label):
    ys = []
    for i in range(label.size):
        tmp = np.zeros(40)
        tmp[label[i] - 1] = 1
        ys.append(tmp)
    return ys


def train(data, label):


    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, SIZE, SIZE, trainSet.NUM_CHANNELS],
                       name='x_input')

    y_ = tf.placeholder(tf.float32, [None, trainSet.OUTPUT_NODE], name='y_output')

    # 使用正则表达式计算损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    min_after_dequeue = 100     # 出队后，队列至少剩下min_after_dequeue个数据

    capacity = min_after_dequeue + 3 * BATCH_SIZE     # 队列的长度
    # shuffle的作用在于指定是否需要随机打乱样本的顺序，一般作用于训练阶段，提高鲁棒性。

    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=BATCH_SIZE,
        capacity=capacity, min_after_dequeue=min_after_dequeue
    )
    y = trainSet.inference(x, False, regularizer)

    # 设定trainable=False 可以防止该变量被数据流图的 GraphKeys.TRAINABLE_VARIABLES 收集
    # 这样我们就不会在训练的时候尝试更新它的值
    global_step = tf.Variable(0, trainable=False)

    # tf.train.ExponentialMovingAverage()函数实现滑动平均模型和计算变量的移动平均值。
    # 衰减率（decay），用于控制模型的更新速度 num_updates提供用来动态设置decay的参数
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVE_AVERAGE_DECAY, global_step
    )

    variable_averages_op = variable_averages.apply(tf.trainable_variables())    # op变量c
    # 交叉熵损失函数
    # logits 必须具有shape [batch_size, num_classes] 并且 dtype (float32 or float64)
    # labels 必须具有shape [batch_size]，并且 dtype int64
    # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引 axis = 1:行
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算损失函数平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)


    # 总的损失
    # tf.add_n() 将list中的数值相加
    # tf.get_collection(‘list_name’)：返回名称为list_name的列表
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 通过tf.train.exponential_decay函数实现指数衰减学习率。1.首先使用较大学习率(目的：为快速得到一个比较优的解);2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
    #  learning_rate为事先设定的初始学习率;decay_rate为衰减系数;decay_steps为衰减速度。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,     # 原始学习率
        global_step,    # 使用指数衰减学习率好比一个计数器，你每进行一次更新它就会增一
        320/BATCH_SIZE,     # 衰减间隔 每隔多少步会更新一次学习率（它只有在staircase为true时才有效）
        LEARNING_RATE_DECAY,    # 衰减率
        staircase=True      # 为true则每隔decay_steps步对学习率进行一次更新
    )

    # 优化损失函数
    # global_step：通常于学习率变化一起使用，可选变量，在变量更新后增加1。
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    # with tf.control_dependencies([a, b]):
    # c= tf.no_op(name='train')#tf.no_op；什么也不做
    # sess.run(c)
    # 确保a，b按顺序都执行。

    # 验证
    with tf.Session() as sess:
        # 必须要使用global_variables_initializer的场合
        # 含有tf.Variable的环境下，因为tf中建立的变量是没有初始化的，也就是在debug时还不是一个tensor量，而是一个Variable变量类型
        tf.global_variables_initializer().run()
        # 实现对Session中多线程的管理：tf.Coordinator和 tf.QueueRunner，这两个类往往一起使用。
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 迭代的神经网络
        for i in range(TRAINING_STEPS):
            xs, ys = sess.run([image_batch, label_batch])
            xs = xs/255.0     # 归一化处理
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, SIZE, SIZE, trainSet.NUM_CHANNELS))
            # 将图像和标签数据通过tf.train.shuffle_batch整理成训练时需要的batch
            ys = get_label(ys)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            if i % 100 == 0:

                # 每10轮输出一次在训练集上的测试结果
                acc = loss.eval({x: reshaped_xs, y_: ys})
                print("After %d training step[s], loss on training"
                      " batch is %g. " % (step, loss_value))


                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step
                )
            coord.request_stop()
            coord.join(threads)





















def main(argv=None):
    # 显示tfrecord格式的图片
    filename_queue = tf.train.string_input_producer(["orl_train.tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    label = tf.cast(features['label'], tf.int32)
    train(img, label)
    """
def main(argv=None):
    # 提取tfrecord格式的文件
    filename_queue = tf.train.string_input_producer(['orl_train.tfrecords'])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)      # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    label = tf.cast(features['label'], tf.int32)
    train(img, label)

"""

if __name__ == '__main__':

    tf.app.run()


































