import numpy
def read_image(filename, start, num):
    image = numpy.zeros((num,28*28)) #创建num行，28*28列的训练ndarray数组
    with open(filename, 'rb') as pf:
        pf.seek(16 + 28 * 28 * start)#定位到第start图片处,16个字节为注释
        for ind in range(num):
            for row in range(28*28):
                data = pf.read(1)#每个灰度值为unsigned byte,将其读取到data里
                pix = int.from_bytes(data, byteorder='big')#读取大端方式存储的数据，将data转为10进制整型数
                if pix > 50:#过滤
                    image[ind][row] = 1
            print('已读取', ind+1, '张图片  ---  ', '剩余图片', num-ind-1, sep='--')
        print('读取结束')
    return image

def read_label(filename, start, num):
    image = numpy.zeros(num) #创建num个数据的标签数组
    with open(filename, 'rb') as pf:
        pf.seek(8 + start)#定位到第start图片的标签处,8个字节为注释
        for ind in range(num):
                data = pf.read(1)
                sign = int.from_bytes(data, byteorder='big')
                image[ind] = sign
                print('已匹配', ind+1, '张图片的标签  ---  ', '剩余图片', num-ind-1, sep='--')
        print('读取结束')
    return image

def classify(test_data,train_data, label_data, K):
    sum = pow(numpy.sum((test_data - train_data) ** 2, axis=1),0.5) #axis为1，用于对行求和，即欧式求和
    l_sort = numpy.zeros(10)  # 存放0~9标签的权重，用于排序
    tip = numpy.argsort(l_sort)
    K = K + 1
    while l_sort[tip[8]] == l_sort[tip[9]]:
        K = K - 1 #如果权重最大的两个标签的权重相等，则减小K的值
        find_arr = numpy.argsort(sum)[:K]#排序后切出前k个最小的数字的下标
        label = label_data[find_arr]
        for i in range(10):
            p = label[label == i].size / K #计算每个数在K中的权重
            l_sort[i] = p
        tip = numpy.argsort(l_sort)
    return tip[9]
#图片集60000个28*28字节图片，测试集10000个28*28字节图片
train_image = 'D:\knndata\\train-images.idx3-ubyte'
train_label = 'D:\knndata\\train-labels.idx1-ubyte'
test_image = 'D:\knndata\\t10k-images.idx3-ubyte'
test_label = 'D:\knndata\\t10k-labels.idx1-ubyte'
start, num =map(int,input('请输入训练起始位置和训练集大小[空格隔开]').split())
train_image_arr = read_image(train_image, start, num)
train_label_arr = read_label(train_label, start, num)
start, num =map(int,input('请输入测试起始位置和测试集大小[空格隔开]').split())
test_image_arr = read_image(train_image, start, num)
test_label_arr = read_label(train_label, start, num)
right = 0
for i in range(num):
    label = classify(test_image_arr[i], train_image_arr , train_label_arr , 5)
    if test_label_arr[i] == label:
        print(i,'RIGHT')
        right = right + 1
    else:
        print(i, 'WRONG!!!!!')
print("已成功识别结束%d张图片  "%(num),'正确率:', (float(right/num) * 100), '%')
