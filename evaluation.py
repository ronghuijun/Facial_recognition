import time
import tensorflow as tf
import numpy as np
import trainSet
import train
import cv2

# 每一秒加载一次最新的模型 在测试数据上测试最新模型的准确率
EVAL_INTERVAL_SECS = 1

def get_label(label):
    ys = []
    for i in range(label.size):
        tmp = np.zeros(40)
        tmp[label[i]-1] = 1
        ys.append(tmp)

    return ys
# tf.Graph() 表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图，通俗来讲就是：在代码中添加的操作（画中的结点）和数据（画中的线条）都是画在纸上的“画”，而图就是呈现这些画的纸，你可以利用很多线程生成很多张图，但是默认图就只有一张。
# tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图，如果只有一个主线程不写也没有关系，tensorflow 里面已经存好了一张默认图，可以使用tf.get_default_graph() 来调用（显示这张默认纸），当有多个线程就可以创造多个tf.Graph()，就是可以有一个画图本，有很多张图纸，这时候就会有一个默认图的概念了。
def evaluate():
    with tf.Graph().as_default() as g:
        # 输出字符串到一个输入管道队列。
        filename_queue = tf.train.string_input_producer(['orl_test.tfrecords'])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string)

                                           })
        # tf.decode_raw函数的意思是将原来编码为字符串类型的变量重新变回来
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [28, 28, 1])
        label = tf.cast(features['label'], tf.int32)
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3*200
        img_batch, label_batch = tf.train.shuffle_batch(
            [img, label], batch_size=80,    # 从队列中提取新的批量大小．
            capacity=capacity, min_after_dequeue=min_after_dequeue
        )

        x = tf.placeholder(
            tf.float32, [80, trainSet.IMG_SIZE, trainSet.IMG_SIZE, trainSet.NUM_CHANNELS], name='x_input')
        y_ = tf.placeholder(
            tf.float32, [None, trainSet.OUTPUT_NODE], name='y_input')
        y = trainSet.inference(x, None, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(
            train.MOVE_AVERAGE_DECAY
        )
        # 通过使用variables_to_restore函数，可以使在加载模型的时候将影子变量直接映射到变量的本身，所以我们在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量。
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            tf.summary.FileWriter("./tmp/summary2", graph=sess.graph)
        while True:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # test = cv2.imread('./data/10/10.jpg')
                # test = cv2.cvtColor(test, cv2.COLOR_BGR2BGRA)
                # test = np.array(test)
                # test = test/255
                # print(test.shape)
                # test_reshape = np.reshape(test, [1, 28, 28, 4])

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                xs, ys = sess.run([img_batch, label_batch])
                ys = get_label(ys)
                xs = xs/255.0
                cpkt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

                if cpkt and cpkt.model_checkpoint_path:
                    # 加载模型

                    saver.restore(sess, cpkt.model_checkpoint_path)     # 下面的restore就是在当前的sess下恢复了所有的变量
                    # 通过文件名得到模型保存时迭代的轮数
                    print(cpkt.model_checkpoint_path)
                    global_step = cpkt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    print("After %s training steps,"
                          "accuracy = %g" % (global_step, accuracy_score))

                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    evaluate()
if __name__ == '__main__':
    tf.app.run()



























