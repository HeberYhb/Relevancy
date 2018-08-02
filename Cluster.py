#!usr/bin/env python
#encoding: utf-8

import pandas as pd
import numpy as np
import jieba
from nltk.tokenize import WordPunctTokenizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
#from gensim.models import Word2Vec

#结巴中文分词
def JieBaSplit(testDt_List):
    seg_Dt_List = []
    for i in range(0, len(testDt_List) - 1):
        seg_Dt=jieba.cut(testDt_List[i])
        seg_Dt_List.append('/'.join(seg_Dt))
    return seg_Dt_List

#nltk英文分词
def nltkSplit(testDt_List):
    seg_Dt1_List = []
    for i in range(0,len(testDt_List)-1):
        seg_Dt1 = WordPunctTokenizer().tokenize(testDt_List[i])
        seg_Dt1_List.append(seg_Dt1)
    return seg_Dt1_List

def toTfidfVec(testDt_List):
    #生成tfidf特征向量
    tfidf_vec = TfidfVectorizer()
    testDtTotfVec = tfidf_vec.fit_transform(testDt_List).todense()
    #print(testDtTotfVec)
    test_arr=np.array(testDtTotfVec)
    #print(test_arr[3][0])
    #print(len(test_arr[0]))
    return test_arr

#生成count特征向量
def toCountVec(testDt_List):
    count_vec=CountVectorizer()
    testDtToVec=count_vec.fit_transform(testDt_List).todense()
    test_arr = np.array(testDtToVec)
    #print(len(test_arr[0]))
    return test_arr

#将生成的特征向量输出成CSV文件
def printToCSV(testDtTotfVec):
    test=pd.DataFrame(testDtTotfVec)
    test.to_csv('./data/LogAlarm_testDttfVec.csv',index=False,header=None)


#pca降维
def DimenReduPCA(test_arr,dimension):
    pca = PCA(n_components=dimension)  # 初始化PCA
    X = pca.fit_transform(test_arr)
    return X

#kmeans聚类  基于划分
def kmeans(test_arr,k,testDt_List):
    km = joblib.load('doc_cluster.pkl')
    y2 = km.fit_predict(test_arr)
    label = []  # 每个样本所属的类
    for i in range(1, len(y2)):
        label.append((testDt_List[i - 1], y2[i - 1]))
        # print(i,cluster.labels_[i-1])
    print(km.inertia_)
    '''
    X=np.array(test_arr)
    markers = ['^', 'x', 'o', '*', '+']
    colors=['b','r','y','g','c']
    for i in range(k):
        members = cluster.labels_ == i
        plt.scatter(X[members, 0], X[members, 1], s=60, marker=markers[i], c=colors[i], alpha=0.5)
    plt.title(' ')
    plt.show()
    '''
    return label

#birch聚类 基于层次
# （可利用其它的一些聚类算法比如K-Means对所有的CF元组进行聚类，得到一颗比较好的CF Tree.这一步的主要目的是消除由于样本读入顺序导致的不合理的树结构，以及一些由于节点CF个数限制导致的树结构分裂）
def birch(test_arr,testDt_List,T,B):
    cluster = Birch(n_clusters=None,threshold=T,branching_factor=B)  #可能需要调threshold参数
    y = cluster.fit_predict(test_arr)
    print(y)
    label = []  # 每个样本所属的类
    for i in range(1, len(cluster.labels_)):
        label.append((testDt_List[i - 1], cluster.labels_[i - 1]))
    return label

#DBSCAN聚类  基于密度
def DBScan(test_arr,r,minPoint,testDt_List):
    cls=DBSCAN(eps=r, min_samples=minPoint)
    y=cls.fit_predict(test_arr)
    print(y)
    label = []  # 每个样本所属的类
    for i in range(0, len(cls.labels_)):
        label.append((testDt_List[i], cls.labels_[i]))
    return label

def Silhouette(test_arr, y):
    silhouette_avg = silhouette_score(test_arr, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(test_arr, y)  # 每个点的轮廓系数
    print(silhouette_avg)
    return silhouette_avg, sample_silhouette_values


if __name__ == '__main__':
    # 取语料库
    testDt_List = []
    with open('./data/LogAlarm_testDt2.csv', encoding='utf-8') as file:
        next(file)
        for line in file:
            testDt = line.split(',')
            testDt_List.append(testDt[12])
            # print(testDt_List)
    dimension=66
    test_arr=toTfidfVec(testDt_List)
    #print(len(test_arr[0]))
    test_arr_lowD= DimenReduPCA(test_arr,dimension)


    k = 30  # 聚类簇数
    label=kmeans(test_arr_lowD,k,testDt_List)

    #B = 30  # birch聚类CF Tree里所有节点的最大CF数
    #T = 0.65  # birch聚类算法中的threshold参数值
    #label=birch(test_arr_lowD,testDt_List,T,B)

    #r=0.7
    #minPoint=3
    #label=DBScan(test_arr_lowD,r,minPoint,testDt_List)

    result=pd.DataFrame(label)
    result.to_csv('./data/LogAlarm_kmeans2_Result.csv',index=False,header=None)
