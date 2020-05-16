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

#色泽 1-3代表 浅白 青绿 乌黑
#根蒂 1-3代表 稍蜷 蜷缩 硬挺
#敲声 1-3代表 清脆 浊响 沉闷
#纹理 1-3代表 清晰 稍糊 模糊
#脐部 1-3代表 平坦 稍凹 凹陷
#触感 1-2代表 硬滑 软粘
#好瓜 1代表 是 0 代表 不是
dataArr, labelArr = createDataSet('西瓜数据集3.0.xlsx')
testVec = [2, 2, 2, 1, 3, 1, 0.697, 0.460]
numSamp = np.shape(dataArr)[0]
numFeat = np.shape(dataArr)[1]
print("Feature num is: ", numFeat)

PC1 = np.sum(labelArr == 1.0)/numSamp
PC0 = np.sum(labelArr == 0.0)/numSamp
print("PC0 = ", PC0)
print("PC1 = ", PC1)
PcondGood = []
PcondBad = []
for i in range(numFeat):
    indexGood = np.where(labelArr == 1.0)
    if i < (numFeat-2):
        PcondGood_i = np.sum(dataArr[indexGood, i] == testVec[i])/np.shape(indexGood)[1]
    else:
        Mean_ci = np.mean(dataArr[indexGood, i])
        Var_ci = np.var(dataArr[indexGood, i], ddof=1)
        PcondGood_i = (1 / (np.sqrt(2 * np.pi * Var_ci)) * np.exp(-(testVec[i] - Mean_ci) ** 2 / (2 * Var_ci)))
    PcondGood.append(PcondGood_i)

for i in range(numFeat):
    indexBad = np.where(labelArr == 0.0)
    if i < (numFeat-2):
        PcondBad_i = np.sum(dataArr[indexBad, i] == testVec[i])/np.shape(indexBad)[1]
    else:
        Mean_ci = np.mean(dataArr[indexGood, i])
        Var_ci = np.var(dataArr[indexGood, i], ddof=1)
        PcondBad_i = (1 / (np.sqrt(2 * np.pi* Var_ci) ) * np.exp(-(testVec[i] - Mean_ci) ** 2 / (2 * Var_ci)))
    PcondBad.append(PcondBad_i)

PBad = cumProduct(PcondBad)*PC0
PGood = cumProduct(PcondGood)*PC1

print("PBad = ", PBad)
print("PGood = ", PGood)
