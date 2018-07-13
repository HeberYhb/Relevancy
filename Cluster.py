#!usr/bin/env python
#encoding: utf-8

import pandas as pd
import numpy as np
import jieba
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from gensim.models import Word2Vec

#取语料库
testDt_List=[]
with open('./data/LogAlarm_testData.csv',encoding='utf-8') as file:
    next(file)
    for line in file:
        testDt=line.split(',')
        testDt_List.append(testDt[12])
#print(testDt_List)

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

#kmeans聚类
def kmeans(test_arr,k):
    clusterer = KMeans(n_clusters=k, init='k-means++')
    y = clusterer.fit_predict(test_arr)
    print(y)
    return y