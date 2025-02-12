import numpy
#读取图片数据
def image_read(filename, start, num):
    image = numpy.zeros((num,28*28)) #image为num行，28*28列的数组，用于保存图片数据
    with open(filename, 'rb') as order:
        order.seek(16 + 28 * 28 * start)#光标到第start张图片处,16个字节为注释
        for hang in range(num):
            for lie in range(28*28):
                data = order.read(1)#每个灰度值为unsigned byte,将其读取到data里
                pixel = int.from_bytes(data, byteorder='big')#以大端方式读取存储的数据，将data转为10进制整型数
                if pixel > 50:#认为灰度值大于50，该点为黑色
                    image[hang][lie] = 1
            print('已读取', hang+1, '张图片  ---  ', '剩余图片', num-hang-1, sep='--')
        print('读取结束')
    return image
#读取标签数据
def label_read(filename, start, num):
    label = numpy.zeros(num) #创建num个数据的标签数组
    with open(filename, 'rb') as order:
        order.seek(8 + start)#定位到第start图片的标签处,8个字节为注释
        for hang in range(num):
                data = order.read(1)
                sign = int.from_bytes(data, byteorder='big')
                label[hang] = sign
                print('已匹配', hang+1, '张图片的标签  ---  ', '剩余图片', num-hang-1, sep='--')
        print('读取结束')
    return label
#对测试图片进行识别
def KNN(test_data,train_data, label_data, K):
    sum = pow(numpy.sum((test_data - train_data) ** 2, axis=1),0.5) #对结果数组的行求和，最后得到该测试数据与各训练数据的距离
    order = numpy.zeros(10)  # 存放0~9标签出现频数的数组
    tip = numpy.argsort(order)
    K = K + 1
    while order[tip[8]] == order[tip[9]]:#判断频数最大的两个标签的出现次数是否相等
        K = K - 1 #如果频数最大的两个标签的出现次数相等，则减小K的值
        min_k = numpy.argsort(sum)[:K]#导出排序后前k个最小的数字的下标
        label = label_data[min_k]
        for i in range(10):
            p = label[label == i].size  #统计标签出现的频数
            order[i] = p
        tip = numpy.argsort(order)#按统计的频数从小到大排序，并导出排序结果的下标
    return tip[9]

#主函数
#图片集60000个28*28字节图片，测试集10000个28*28字节图片
train_image = 'D:\knndata\\train-images.idx3-ubyte'   #训练集及测试集的路径
train_label = 'D:\knndata\\train-labels.idx1-ubyte'
test_image = 'D:\knndata\\t10k-images.idx3-ubyte'
test_label = 'D:\knndata\\t10k-labels.idx1-ubyte'
start, num =map(int,input('请输入训练起始位置和训练集大小[空格隔开]').split())
train_image_array = image_read(train_image, start, num) #读取训练数据
train_label_array = label_read(train_label, start, num) #得去训练数据图片的标签
start, num =map(int,input('请输入测试起始位置和测试集大小[空格隔开]').split())
test_image_array = image_read(test_image, start, num) #读取测试数据
test_label_array = label_read(test_label, start, num) #读取测试数据图片的标签
right = 0
K = eval(input('请输入K值：'))
for i in range(num):
    label = KNN(test_image_array[i], train_image_array , train_label_array , K)
    if test_label_array[i] == label:
        print(i,'recognition right')
        right = right + 1
    else:
        print(i, 'recognition wrong!!!')
print("成功识别%d张图片  "%(num),'正确率:', (float(right/num) * 100), '%')
