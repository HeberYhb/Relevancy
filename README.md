# Relevancy
故障日志分析: 使用关联算法和文本聚类算法将同一时间段相同的报警日志关联
./data/LogAlarm_testData.csv 是原始报警日志数据
./data/LogAlarm_testDttfVec.csv 是使用sklearn的TfidfVectorizer获取的特征向量
./data/LogAlarm_testDtVec.csv   是使用sklearn的CountVectorizer获取的特征向量
./data/LogAlarm_kmeans_Result.csv 是kmeans聚类结果

./Relevancy文件夹下是FP-Growth和Apriori类，将数据按天分成集合作为输入

