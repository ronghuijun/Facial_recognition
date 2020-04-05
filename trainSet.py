import tensorflow as tf
# 定义CNN网络结构的参数

INPUT_NODE = 784     # 输入层节点
OUTPUT_NODE = 40    # 输出层节点

IMG_SIZE = 28
NUM_CHANNELS = 1    # 图片通道数
NUM_LABELS = 40

# 卷积层C1 6个
CONV1_DEEP = 6
CONV1_SIZE = 5

# 卷积层C3 10个
CONV2_DEEP = 10
CONV2_SIZE = 5

# 全连接层输出
FC_SIZE = 160

# TF中有两种作用域类型：命名域 (name scope)，通过tf.name_scope 或 tf.op_scope创建；
# 变量域 (variable scope)，通过tf.variable_scope 或 tf.variable_op_scope创建；
# 这两种作用域，对于使用tf.Variable()方式创建的变量，具有相同的效果，都会在变量名称前面，加上域名称。
# 对于通过tf.get_variable()方式创建的变量，只有variable scope名称会加到变量名称前面，而name scope不会作为前缀。

# CNN网络结构
def inference(input_tensor, train, regularizer):
    # 第一层卷积层
    # TensorFlow提供了Variable Scope 这种独特的机制来共享变量。
    with tf.variable_scope('layer1_conv1'):
        # tf.get_variablef.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式
        # tf.truncated_normal_initializer：截取的正态分布

        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
            # tf.truncated_normal_initializer从截断的正态分布中输出随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
            # stddev：一个python标量或一个标量张量。要生成的随机值的标准偏差。
        )
        conv1_biases = tf.get_variable(
            # tf.constant_initializer初始化为常数，这个非常有用，通常偏置项就是用它初始化的。
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID'
        )
        #  tf.nn.relu这个函数的作用是计算激活函数 relu，即 max(features, 0)。
        #  将大于0的保持不变，小于0的数置为0。
        # tf.nn.bias_add一个叫bias的向量加到一个叫value的矩阵上，是向量与矩阵的每一行进行相加，得到的结果和value矩阵大小相同。
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.avg_pool(
            # ksize 池化窗口的大小
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'
        )

    # 第三层卷积层
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable(
            'weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            'bias', [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        # 使用边长为5，深度为10的过滤器，过滤器移动的步长为1
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化层
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.avg_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'
        )
        # 讲第四层的输出转化为向量
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五层 全连接层
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable(
            'weight', [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 加入正规化
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE], initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            # tf.nn.dropout()是为了防止或减轻过拟合而使用的函数，它一般用在全连接层
            fc1 = tf.nn.dropout(fc1, 0.5) # 0.5表示每个元素被保留下来的概率
    # 第六层 全连接层
    with tf.variable_scope('layer6_fc2'):
        fc2_weights = tf.get_variable(
            'weight', [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias', [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.add(tf.matmul(fc1, fc2_weights), fc2_biases)

    return logit












