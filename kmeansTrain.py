#coding=utf8
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import Cluster

def readData(Path):
    testDt_List = []
    with open(Path,'r',encoding='utf-8') as file:
        next(file)
        for line in file:
            testDt = line.split(',')
            testDt_List.append(testDt[12])
    dimension = 66
    tfidf_vec = Cluster.toTfidfVec(testDt_List)
    # print(len(test_arr[0]))
    tfidf_vec_lowD = Cluster.DimenReduPCA(tfidf_vec, dimension)
    return tfidf_vec_lowD, testDt_List

tfidf_vec_lowD, testDt_List = readData('./data/LogAlarm_testData.csv')
tfidf_vec2_lowD, testDt_List2= readData('./data/LogAlarm_testDt2.csv')


k=30
km = KMeans(n_clusters=k,init='k-means++')
y = km.fit(tfidf_vec_lowD)
print(y)
joblib.dump(km,  'doc_cluster.pkl')
