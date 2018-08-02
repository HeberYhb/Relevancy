import Relevancy.fpGrowth as fp
#import fpGrowth as fp

testDt_List =fp.loadSimpDat()
print(testDt_List)
#print(simpDat)

initSet = fp.createInitSet(testDt_List)
myFPtree, myHeaderTab = fp.createTree(initSet, 3)

freqItems = []
#print('zidian:',myHeaderTab['r'][0])
fp.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
print('freqItems:',freqItems)