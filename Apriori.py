import Relevancy.fpGrowth as fp
#import apriori as ap
import Relevancy.apriori as ap

testDt_List =fp.loadSimpDat()

C1=ap.createC1(testDt_List)
L,supportData=ap.apriori(testDt_List, 0.1)
print(L)
brl=ap.generateRules(L, supportData, 0.7)
print('brl:',brl)


#如果支持度计算使用文本相似性，然后频繁集？