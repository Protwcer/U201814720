# -*- coding: UTF-8 -*-
import numpy
import random
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
'''
函数说明：从.csv文件顺序读取数据和标签值
        标签值：0-男；1-女
'''
def read_data_and_label(filename):
    data = []
    data_single = []
    label = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        sign = 0
        for row in csv_reader:
            if sign == 0:
                sign = sign + 1
            else:
                for num in range(20):
                    data_single.append(float(row[num]))
                dt = data_single.copy()
                data.append(dt)
                data_single.clear()
                if row[20] == 'male':
                    label.append(0)
                else:
                    label.append(1)
    return data,label

'''
函数说明：从读取出的数据里按7:3的比例随机挑选训练集、测试集
'''
def pick(data,label):
    trainingSet = list(range(len(label)))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    train_data = []
    train_label_data =[]
    test_data =[]
    test_label_data =[]
    for i in range(int(len(label)*0.3/2)):
        randIndex_m = random.randint(0, int(len(trainingSet)/2)-1)  # 随机选取索引值
        randIndex_w = random.randint(int(len(trainingSet)/2), len(trainingSet)-2)  # 随机选取索引值
        testSet.append(trainingSet[randIndex_m])  # 添加测试集的索引值
        testSet.append(trainingSet[randIndex_w])  # 添加测试集的索引值
        del (trainingSet[randIndex_m])  # 在训练集列表中删除添加到测试集的索引值
        del (trainingSet[randIndex_w])  # 在训练集列表中删除添加到测试集的索引值
    for docIndex in trainingSet:
        train_data.append(data[docIndex])
        train_label_data.append(label[docIndex])
    for docIndex in testSet:
        test_data.append(data[docIndex])
        test_label_data.append(label[docIndex])
    return train_data,train_label_data,test_data,test_label_data

'''
函数说明：高斯朴素贝叶斯分类器训练函数,返回不同标签下个分量的均值和方差
'''
def trainGS(train_data,label_data):
    num_man = num_woman = 0
    average_m = numpy.zeros(20)
    average_w = numpy.zeros(20)
    variance_m = numpy.zeros(20)
    variance_w = numpy.zeros(20)
    for i in range(len(label_data)):
        if label_data[i] == 0:
            average_m += train_data[i]
            num_man = num_man + 1
        else:
            average_w += train_data[i]
            num_woman = num_woman + 1
    average_m = average_m/num_man
    average_w = average_w/num_woman
    for i in range(len(label_data)):
        if label_data[i] == 0:
            variance_m += numpy.power(train_data[i]-average_m,2)
        else:
            variance_w += numpy.power(train_data[i]-average_w,2)
    variance_m = variance_m/num_man
    variance_w = variance_w/num_woman
    return average_m,variance_m,average_w,variance_w

'''
函数说明：高斯朴素贝叶斯分类器
'''
def classifyGS(tset_data,average_m,variance_m,average_w,variance_w):
    p_m = numpy.zeros(20)
    p_w = numpy.zeros(20)
    s0 = 1 / numpy.power(variance_m,0.5)
    s1 = 1 / numpy.power(variance_w,0.5)
    p_m = numpy.log(s0) - numpy.power(tset_data - average_m, 2) / (2 * variance_m)
    p_w = numpy.log(s1) - numpy.power(tset_data - average_w, 2) / (2 * variance_w)
    f0 = numpy.sum(p_m)
    f1 = numpy.sum(p_w)
    if f0 < f1:
        return 1
    else:
        return 0

'''
函数说明：KNN分类器
'''
def KNN(test_data,train_data,train_label_data,K,rd):
    sum = numpy.linalg.norm(test_data - train_data,ord=rd,axis=1) #欧式距离
    order = numpy.zeros(2)  # 存放标签0、1出现频数的数组
    tip = numpy.argsort(order)
    K = K + 1
    while order[tip[0]] == order[tip[1]]:  # 判断两个标签的出现次数是否相等
        K = K - 1  # 如果频数最大的两个标签的出现次数相等，则减小K的值
        min_k = numpy.argsort(sum)[:K]  # 导出排序后前k个最小的数字的下标
        label = train_label_data[min_k]
        for i in range(2):
            p = label[label == i].size  # 统计标签出现的频数
            order[i] = p
        tip = numpy.argsort(order)  # 按统计的频数从小到大排序，并导出排序结果的下标
    return tip[1]
'''
函数说明：随机梯度下降法进行训练
'''
def SGD_train(train_data,train_label_data,a):
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    clf = SGDClassifier(alpha=a,loss='hinge' ,penalty='l2')
    clf.fit(train_data,train_label_data)
    return clf
'''
函数说明：逻辑回归训练
'''
def LOG_train(train_data,train_label_data):
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    lr = LogisticRegression()
    lr.fit(train_data,train_label_data)
    return lr

def draw_a(data,label):
    a_X = []
    RIGHT_Y = []
    right_number_SDG = 0
    for i in range(1,100):
        a_X.append(i)
    for a in a_X:
        for x in range(3):
            train_data,train_label_data,test_data,test_label_data = pick(data,label)
            leng = len(test_data)
            clf = SGD_train(numpy.array(train_data), numpy.array(train_label_data),a)
            ss = StandardScaler()
            test_data1 = test_data.copy()
            test_data1 = ss.fit_transform(test_data1)
            predict_label_SGD = clf.predict(numpy.array(test_data1))
            for i in range(len(predict_label_SGD)):
                if predict_label_SGD[i] == test_label_data[i]:
                    right_number_SDG += 1
            train_data.clear()
            train_label_data.clear()
            test_data.clear()
            test_label_data.clear()
        RIGHT_Y.append(right_number_SDG / (leng * 3))
        right_number_SDG = 0
    plt.plot(a_X, RIGHT_Y, linewidth=1.5, c='blue')
    plt.xlabel('α', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    # 设置刻度数字的大小
    plt.tick_params(axis='both', labelsize=10)
    plt.show()

def draw_K(data,label):
    high = eval(input('请输入K值上限：'))
    K_X = []
    RIGHT_Y = []
    right_number_KNN = 0
    clo = ['red','blue','yellow']
    rod = [1,2,numpy.inf]
    rods = ['1','2','inf']
    for r in range(3):
        for K in range(1,high+1,2):
            K_X.append(K)
            for x in range(3):
                train_data, train_label_data, test_data, test_label_data = pick(data, label)
                leng = len(test_data)
                for i in range(len(test_data)):
                    c = KNN(numpy.array(test_data[i]),numpy.array(train_data),numpy.array(train_label_data),K,rod[r])
                    if c == test_label_data[i]:
                        right_number_KNN += 1
                train_data.clear()
                train_label_data.clear()
                test_data.clear()
                test_label_data.clear()
            RIGHT_Y.append(right_number_KNN/(leng*3))
            right_number_KNN = 0
        RIGHT_Yc = RIGHT_Y.copy()
        K_Xc = K_X.copy()
        RIGHT_Y.clear()
        K_X.clear()
        plt.plot(K_Xc, RIGHT_Yc, linewidth=1.5,c = clo[r],label=rods[r])
    plt.xlabel('K', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    # 设置刻度数字的大小
    plt.legend()
    plt.tick_params(axis='both', labelsize=10)
    plt.show()

if __name__ == '__main__':
    filename = 'D:\\voice.csv'
    data,label = read_data_and_label(filename)
    train_data,train_label_data,test_data,test_label_data = pick(data,label)
    right_number_Bayes = right_number_KNN = right_number_SDG = right_number_LOG = 0
    #average_m,variance_m,average_w,variance_w = trainGS(numpy.array(train_data),numpy.array(train_label_data))
    print('------------------------------------------------------')
    K = 1
    for i in range(len(test_data)):
        c = KNN(numpy.array(test_data[i]),numpy.array(train_data),numpy.array(train_label_data),K,1)
        if c == test_label_data[i]:
            #rint('第',i+1,'个音频：Right!')
            right_number_KNN += 1
        #else:
            #print('第',i+1,'个音频：Wrong!!!')
    print('Bayes:准确率：', float(right_number_Bayes / len(test_data)) * 100,'%')
    print('KNN:准确率：', float(right_number_KNN / len(test_data)) * 100,'%  K =',K)
    #draw_K(data,label)
    draw_a(data,label)
    clf = SGD_train(numpy.array(train_data),numpy.array(train_label_data),0.0001)
    ss = StandardScaler()
    test_data = ss.fit_transform(test_data)
    predict_label_SGD = clf.predict(numpy.array(test_data))
    for i in range(len(predict_label_SGD)):
        if predict_label_SGD[i] == test_label_data[i]:
            right_number_SDG += 1
    print('SGD:准确率：', float(right_number_SDG / len(test_data)) * 100, '%')
    res = clf.score(numpy.array(test_data),numpy.array(test_label_data))
    print(res)
    lr = LOG_train(numpy.array(train_data), numpy.array(train_label_data))
    predict_label_LOG = lr.predict(numpy.array(test_data))
    for i in range(len(predict_label_LOG)):
        if predict_label_LOG[i] == test_label_data[i]:
            right_number_LOG += 1
    print('LOG:准确率：', float(right_number_LOG / len(test_data)) * 100, '%')
