import xlrd
import numpy as np

# 一维数组求累乘
def cumProduct(arr):
    result = 1
    for i in range(len(arr)):
        result *= arr[i]

    return result


def createDataSet(filename):
    workbook = xlrd.open_workbook(filename)
    name = workbook.sheet_names()
    print("workbook name: ", name)
    worksheet = workbook.sheet_by_index(0)
    print("worksheet is: ", worksheet)

    nrows = worksheet.nrows
    print("rows number is: ", nrows)
    ncols = worksheet.nrows
    print("cols number is: ", ncols)

    dataArr = []
    labelArr = []
    for i in range(1, nrows):
        rowVals = np.array(worksheet.row_values(i))
        dataArr.append(rowVals[1:-1])
        labelArr.append(rowVals[-1])

    return np.array(dataArr), np.array(labelArr)


def splitDataSet(dataArr, labelArr, K=10, k=0):
    trainingData = [x for i, x in enumerate(dataArr) if i % K != k]
    trainingLabel = [x for i, x in enumerate(labelArr) if i % K != k]
    testData = [x for i, x in enumerate(dataArr) if i % K == k]
    testLabel = [x for i, x in enumerate(labelArr) if i % K == k]
    trainingData = np.array(trainingData)
    trainingLabel = np.array(trainingLabel)
    testData = np.array(testData)
    testLabel = np.array(testLabel)
    return trainingData, trainingLabel, testData, testLabel

def NaiveBayes(dataArr, labelArr, testVec):
    numSamp = np.shape(dataArr)[0]
    numFeat = np.shape(dataArr)[1]
    #print("Sample num is: ", numSamp)
    #print("Feature num is: ", numFeat)
    # 计算先验概率P(ci)
    #labelN = len(set(labelArr))
    labelUniq = np.unique(labelArr)
    labelN = len(np.unique(labelArr))
    PC = []         # 保存PC值列表
    for i in range(labelN):
        PC.append( (np.sum(labelArr == labelUniq[i])+1) / (numSamp + labelN) )
    PC = np.array(PC)
    #print(PC)
    Pcond_Feat = np.zeros(labelN)
    Pcond = np.ones_like(labelN)

    # 对每一个xi，计算类条件概率P(xi|cj)，然后累乘得到P(x|cj)
    # 先计算P(xi|c1)，并将结果保存到PcondGood列表中
    for i in range(numFeat):
        for j in range(labelN):
            indexClass = np.where(labelArr == labelUniq[j])
            attrN = len(np.unique(dataArr[:, i]))
            if attrN < 5:               # 离散值使用大数定理进行估计
                #print(np.sum(dataArr[indexClass, i] == testVec[i]))
                #print(len(indexClass[0]))
                Pcond_Feat[j] = (np.sum(dataArr[indexClass, i] == testVec[i])+1)/(len(indexClass[0]) + attrN)    # laplace平滑
            else:                       # 连续值先假设服从高斯分布，然后采用极大似然估计
                Mean_cij = np.mean(dataArr[indexClass, i])
                Var_cij = np.var(dataArr[indexClass, i], ddof=1)
                Pcond_Feat[j] = (1 / (np.sqrt(2 * np.pi * Var_cij)) * np.exp(-(testVec[i] - Mean_cij) ** 2 / (2 * Var_cij)))
        Pcond = np.multiply(Pcond, Pcond_Feat)


    # 计算联合概率P(x|ci)
    Puni = Pcond*PC
    #print("Puni = ", Puni)
    indBest = np.argmax(Puni)
    #print("The answer is: ", labelUniq[indBest])
    return labelUniq[indBest]

#色泽 1-3代表 浅白 青绿 乌黑
#根蒂 1-3代表 稍蜷 蜷缩 硬挺
#敲声 1-3代表 清脆 浊响 沉闷
#纹理 1-3代表 清晰 稍糊 模糊
#脐部 1-3代表 平坦 稍凹 凹陷
#触感 1-2代表 硬滑 软粘
#好瓜 1代表 是 0 代表 不是
#dataArr, labelArr = createDataSet('西瓜数据集3.0.xlsx')
#testVec = [2, 2, 1, 1, 3, 1, 0.697, 0.460]      # 敲声为清脆，需要用到laplace平滑
#dataArr, labelArr = createDataSet('婚姻分析.xlsx')
#testVec = [1, 1, 2, 1]      # 敲声为清脆，需要用到laplace平滑
'''
| class values
unacc, acc, good, vgood
| attributes
buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.
'''
dataArr, labelArr = createDataSet('./car_data/car.xlsx')
K = 10  # 10-fold verification
for k in range(K):
    trainingData, trainingLabel, testData, testLabel = splitDataSet(dataArr, labelArr, K, k)
    numTest = len(testLabel)
    numRight = 0
    for i in range(len(testLabel)):
        testVec = testData[i]
        classReturn = NaiveBayes(trainingData, trainingLabel, testVec)
        if (classReturn == testLabel[i]):
            numRight += 1
    accuracy = numRight / numTest
    print(k, "-th Classification accuracy is: ", accuracy)



