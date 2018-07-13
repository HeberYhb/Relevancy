#!usr/bin/env python
#encoding: utf-8

import pandas as pd
import jieba
from sklearn.feature_extraction import dict_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from gensim.models import Word2Vec

testDt_List=[]
with open('./data/LogAlarm_testData.csv',encoding='utf-8') as file:
    next(file)
    for line in file:
        testDt=line.split(',')
        testDt_List.append(testDt[12])
#print(testDt_List)


#分词
for i in range(0,len(testDt_List)-1):
    seg_Dt=jieba.cut(testDt_List[i],cut_all=True)
    print('/'.join(seg_Dt))

'''
#生成特征向量
count_vec=CountVectorizer()
testDtToVec=count_vec.fit_transform(testDt_List).todense()
'''

'''
#outfile=open('./data/LogAlarm_testDtVec.csv','w')
test=pd.DataFrame(testDtToVec)
test.to_csv('./data/LogAlarm_testDtVec.csv',index=False,header=None)
'''


