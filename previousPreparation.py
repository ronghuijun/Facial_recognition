# ORL训练集一共有40类，每一类有10张bmp类型的图片
# 将这些数据读入，制作训练集和测试集
import os
import cv2
import random


input_path = './orl'
train_path = './train'
test_path = './test'
if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)

for i in range(1, 41):
    if not os.path.exists(train_path + '/' + str(i)):
        os.mkdir(train_path + '/' + str(i))
    if not os.path.exists(test_path + '/' + str(i)):
        os.mkdir(test_path + '/' + str(i))



def generateData(train_path, test_path):
    i = 1
    output_index = 1
    # input_path 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    for(root, dirs, files) in os.walk(input_path):
        # 随机选8张做训练集 2张测试
        random.shuffle(files)
        for file in files:
            if file.endswith('.bmp'):

                img_path = root + '/' + file
                # openCV读图片
                img_data = cv2.imread(img_path)
                # 按照论文把图片调节28*28
                # INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
                img_data = cv2.resize(img_data, (28, 28), interpolation=cv2.INTER_AREA)
                if i <= 2:
                    print(test_path + '/' + str(output_index) + '/' + str(i) + '.jpg')
                    cv2.imwrite(test_path + '/' + str(output_index) + '/' + str(i) + '.jpg', img_data)
                    i += 1
                elif 10>=i and i>=3:
                    cv2.imwrite(train_path + '/' + str(output_index) + '/' + str(i) + '.jpg', img_data)
                    i += 1
                if i > 10:
                    output_index += 1
                    i = 1

generateData(train_path, test_path)
