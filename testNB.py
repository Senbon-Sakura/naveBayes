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

    print(dataArr)
    print(labelArr)
    return np.array(dataArr), np.array(labelArr)

def NaiveBayes(dataArr, labelArr, testVec):
    numSamp = np.shape(dataArr)[0]
    numFeat = np.shape(dataArr)[1]
    print("Sample num is: ", numSamp)
    print("Feature num is: ", numFeat)
    # 计算先验概率P(ci)
    #labelN = len(set(labelArr))
    labelN = len(np.unique(labelArr))
    PC1 = (np.sum(labelArr == 1.0)+1)/(numSamp + labelN)        # laplace平滑
    PC0 = (np.sum(labelArr == 0.0)+1)/(numSamp + labelN)        # laplace平滑
    print("PC0 = ", PC0)
    print("PC1 = ", PC1)

    PcondGood = 1
    PcondBad = 1
    # 对每一个xi，计算类条件概率P(xi|cj)，然后累乘得到P(x|cj)
    # 先计算P(xi|c1)，并将结果保存到PcondGood列表中
    for i in range(numFeat):
        indexGood = np.where(labelArr == 1.0)
        attrN = len(np.unique(dataArr[:, i]))
        if attrN < 5:               # 离散值使用大数定理进行估计
            PcondGood_i = (np.sum(dataArr[indexGood, i] == testVec[i])+1)/(np.shape(indexGood)[1] + attrN)    # laplace平滑
        else:                       # 连续值先假设服从高斯分布，然后采用极大似然估计
            Mean_ci = np.mean(dataArr[indexGood, i])
            Var_ci = np.var(dataArr[indexGood, i], ddof=1)
            PcondGood_i = (1 / (np.sqrt(2 * np.pi * Var_ci)) * np.exp(-(testVec[i] - Mean_ci) ** 2 / (2 * Var_ci)))
        PcondGood *= PcondGood_i

    # 先计算P(xi|c0)，并将结果保存到PcondBad
    for i in range(numFeat):
        indexBad = np.where(labelArr == 0.0)
        attrN = len(np.unique(dataArr[:, i]))
        if attrN < 5:               # 离散值使用大数定理进行估计
            attrN = len(np.unique(dataArr[:, i]))
            PcondBad_i = (np.sum(dataArr[indexBad, i] == testVec[i]) + 1)/(np.shape(indexBad)[1] + attrN)
        else:                       # 连续值先假设服从高斯分布，然后采用极大似然估计
            Mean_ci = np.mean(dataArr[indexGood, i])
            Var_ci = np.var(dataArr[indexGood, i], ddof=1)
            PcondBad_i = (1 / (np.sqrt(2 * np.pi* Var_ci) ) * np.exp(-(testVec[i] - Mean_ci) ** 2 / (2 * Var_ci)))
        PcondBad *= PcondBad_i

    # 计算联合概率P(x|ci)
    PBad = PcondBad*PC0
    PGood = PcondGood*PC1

    print("PBad = ", PBad)
    print("PGood = ", PGood)

#色泽 1-3代表 浅白 青绿 乌黑
#根蒂 1-3代表 稍蜷 蜷缩 硬挺
#敲声 1-3代表 清脆 浊响 沉闷
#纹理 1-3代表 清晰 稍糊 模糊
#脐部 1-3代表 平坦 稍凹 凹陷
#触感 1-2代表 硬滑 软粘
#好瓜 1代表 是 0 代表 不是
#dataArr, labelArr = createDataSet('西瓜数据集3.0.xlsx')
#testVec = [2, 2, 1, 1, 3, 1, 0.697, 0.460]      # 敲声为清脆，需要用到laplace平滑
dataArr, labelArr = createDataSet('婚姻分析.xlsx')
testVec1 = [0, 0, 0, 0]
testVec2 = [1, 1, 2, 1]

NaiveBayes(dataArr, labelArr, testVec1)
NaiveBayes(dataArr, labelArr, testVec2)

