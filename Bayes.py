# -*- coding: UTF-8 -*-
import nltk
import numpy
import re
import random
import csv
"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)               #创建一个其中所含元素都为0的向量
    for word in inputSet:                          #遍历每个词条
        if word in vocabList:                      #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec        #返回文档向量


"""
函数说明:根据vocabList词汇表，构建词袋模型
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:             # 遍历每个词条
        if word in vocabList:         # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec  # 返回词袋模型


"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 正常邮件类的条件概率数组
    p1Vect - 垃圾邮件类的条件概率数组
    pAbusive - 文档属于垃圾邮件类的概率
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = numpy.sum(trainCategory) / float(numTrainDocs)  # 文档属于垃圾邮件类的概率
    p0Num = numpy.ones(numWords)
    p1Num = numpy.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1,拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  # 分母初始化为2 ,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += numpy.sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += numpy.sum(trainMatrix[i])
    p1Vect = numpy.log(p1Num / p1Denom)
    p0Vect = numpy.log(p0Num / p0Denom)   #取对数，防止下溢出
    return p0Vect, p1Vect, pAbusive  # 返回属于正常邮件类的条件概率数组，属于侮辱垃圾邮件类的条件概率数组，文档属于垃圾邮件类的概率


"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 正常邮件类的条件概率数组
	p1Vec - 垃圾邮件类的条件概率数组
	pClass1 - 文档属于垃圾邮件的概率
Returns:
	0 - 属于正常邮件类
	1 - 属于垃圾邮件类
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1  # 对应元素相乘
    #p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
    p1=sum(vec2Classify*p1Vec)+numpy.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+numpy.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
函数说明:接收一个大字符串并将其解析为字符串列表，同时剔除停用词和无用词
"""
def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\W+', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    for tok in listOfTokens:
        if len(tok) == 1:
            listOfTokens.remove(tok)
        else:
            tok.lower()
    del(listOfTokens[0])
    filtered_sentence = []
    for w in listOfTokens:  #去停用词
        if w not in stopwords:
            filtered_sentence.append(w)
    return filtered_sentence
'''
函数说明：根据停用词文本得到停用词列表
'''
def to_stopwords(stopword):
    listOfTokens = re.split(r'\W+', stopword)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return listOfTokens #返回停用词列表

"""
函数说明:测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证
"""
def spamTest():
    docList = []
    classList = []
    with open('D:\email_data\spam_ham_dataset.csv') as csv_file:    #读取.csv文件  有5271封邮件
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        start = eval(input('请输入起始邮件位置（最小1）：'))
        end = eval(input('请输入结束邮件位置：'))
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if line_count>=start and line_count<=end :
                    wordList = textParse(row[2])    # 读取每个邮件，并字符串转换成字符串列表
                    docList.append(wordList)
                    classList.append(int(row[3]))    # 记录读取邮件的标记
                line_count += 1
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet = list(range(end-start))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(int((end-start)*0.15)):  # 所选邮件数据里15%作为测试数据
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  # 将生成的词袋模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  # 测试集的词袋模型
        if classifyNB(numpy.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    stopword = open('D:\stopwords.txt', 'r').read()
    global stopwords
    stopwords = to_stopwords(stopword)  #停用词列表
    spamTest()

