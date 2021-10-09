#!/usr/bin/env python
# coding=utf-8

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator



def createDataSet():
	dataSet = [[0, 0, 0, 0, 'no'],						
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']		
	return dataSet, labels


def createTree(dataset,labels,featLabels):
	classList = [example[-1] for example in dataset]		# 标签
	if classList.count(classList[0]) == len(classList):		# number of "no" or "yes"
		return classList[0]									# 全是同一类
	if len(dataset[0]) == 1:								# 只有标签(说明没有更多信息帮助分类, 预测最大可能性的标签返回)
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataset)			# 求使熵最小的feature(使分类效果最好的)
	bestFeatLabel = labels[bestFeat]
	featLabels.append(bestFeatLabel)
	myTree = {bestFeatLabel:{}}
	del labels[bestFeat]
	featValue = [example[bestFeat] for example in dataset]
	uniqueVals = set(featValue)
	for value in uniqueVals:	# 对选出来的feature划分的每个类构造一棵子决策树
		sublabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,bestFeat,value),sublabels,featLabels)
	return myTree

# 选出列中出现最多的元素(返回可能性最大的标签)
def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedclassCount[0][0]

# 使信息增益最大的feature(-1为不再分类熵最小)
def chooseBestFeatureToSplit(dataset):
	numFeatures = len(dataset[0]) - 1		# feature数
	baseEntropy = calcShannonEnt(dataset)	# 原熵值(没分类时)
	bestInfoGain = 0						# 最佳信息增益值
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataset]	# dataset中该feature取值
		uniqueVals = set(featList)
		newEntropy = 0
		for val in uniqueVals:							# 求按该feature分类后熵
			subDataSet = splitDataSet(dataset,i,val)
			prob = len(subDataSet)/float(len(dataset))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i	
	return bestFeature						# 最好的feature
			
			
			
# 取dataset里第axis个标签为val的, 去掉axis, 作为subDataset返回
def splitDataSet(dataset,axis,val):
	retDataSet = []
	for featVec in dataset:
		if featVec[axis] == val:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet
			
# 算熵
def calcShannonEnt(dataset):
	numexamples = len(dataset)
	labelCounts = {}
	for featVec in dataset:
		currentlabel = featVec[-1]
		if currentlabel not in labelCounts.keys():
			labelCounts[currentlabel] = 0
		labelCounts[currentlabel] += 1
		
	shannonEnt = 0
	for key in labelCounts:
		prop = float(labelCounts[key])/numexamples
		shannonEnt -= prop*log(prop,2)
	return shannonEnt


def getNumLeafs(myTree):
	numLeafs = 0												
	firstStr = next(iter(myTree))								
	secondDict = myTree[firstStr]								
	for key in secondDict.keys():
	    if type(secondDict[key]).__name__=='dict':				
	        numLeafs += getNumLeafs(secondDict[key])
	    else:   numLeafs +=1
	return numLeafs


def getTreeDepth(myTree):
	maxDepth = 0												
	firstStr = next(iter(myTree))								
	secondDict = myTree[firstStr]								
	for key in secondDict.keys():
	    if type(secondDict[key]).__name__=='dict':				
	        thisDepth = 1 + getTreeDepth(secondDict[key])
	    else:   thisDepth = 1
	    if thisDepth > maxDepth: maxDepth = thisDepth			
	return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	arrow_args = dict(arrowstyle="<-")											
	font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)		
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',	
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]																
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")										
	leafNode = dict(boxstyle="round4", fc="0.8")											
	numLeafs = getNumLeafs(myTree)  														
	depth = getTreeDepth(myTree)															
	firstStr = next(iter(myTree))																								
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)	
	plotMidText(cntrPt, parentPt, nodeTxt)													
	plotNode(firstStr, cntrPt, parentPt, decisionNode)										
	secondDict = myTree[firstStr]															
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD										
	for key in secondDict.keys():								
		if type(secondDict[key]).__name__=='dict':											
			plotTree(secondDict[key],cntrPt,str(key))        								
		else:																														
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')													#创建fig
	fig.clf()																				#清空fig
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    							#去掉x、y轴
	plotTree.totalW = float(getNumLeafs(inTree))											#获取决策树叶结点数目
	plotTree.totalD = float(getTreeDepth(inTree))											#获取决策树层数
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;								#x偏移
	plotTree(inTree, (0.5,1.0), '')															#绘制决策树
	plt.show()




if __name__ == '__main__':
	dataset, labels = createDataSet()
	featLabels = []
	myTree = createTree(dataset,labels,featLabels)
	createPlot(myTree)

