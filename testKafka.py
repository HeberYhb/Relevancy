from kafka import KafkaConsumer
import json

import Cluster
import pandas as pd
import pymysql
import datetime
import time

'''
client=KafkaClient(hosts="80.2.238.230:2181,80.2.238.230:2182,80.2.238.230:21832")
topic=client.topics['newlog2']

consumer=topic.get_balanced_consumer(
    zookeeper_connect='80.2.238.230:2181,80.2.238.230:2182,80.2.238.230:2183'
)
'''
#连接kafka并消费数据
def KafkaCon():
    #KafkaConsumer的参数：topic='newlog2'为生产者数据是设定的主题;bootstrap_servers是配置kafka服务的接口;
    #consumer_timeout_ms是指间隔多少ms没有新的消息即自动终止接收；auto_offset_reset指从哪个位置开始消费数据，一般实时的都设置成latest
    consumer=KafkaConsumer('newlog2',
                           bootstrap_servers='80.2.238.230:9092',
                           #enable_auto_commit=True,
                           consumer_timeout_ms=1000,
                           #auto_commit_interval_ms=5000,
                           auto_offset_reset='earliest')

    alarmLog=[]
    testDt_List=[]
    i=0
    for message in consumer:
        #print(message.value)
        if(message is not None):
            #取出kafka消费的数据中的value值并将其转为utf-8格式的字符串
            messages=message.value.decode('utf-8')
            #转成json格式
            al=json.loads(messages)
            #取出value中的message信息，即为报警日志所有字段
            result=al['message'].split('||')
            #print(result)
            
            #将时间戳格式转为utc格式
            year=datetime.datetime.now().strftime('%Y')  #取出当前时间的年份
            GMT_FORMAT='%b %d %H:%M:%S %Y'   #%b 英文缩写的月份，%d 哪一天，%H 24小时制的小时，%M 分钟，%S 秒，%Y年份
            times=datetime.datetime.strptime(result[0][0:15]+' '+year,GMT_FORMAT) #将GMT_FORMAT格式的时间戳转为标准的datetime格式
            utcTime=int(time.mktime(datetime.datetime.timetuple(times))) #将datetime转为utc
            result.append(utcTime)
            result[0]=result[0][15:]
            alarmLog.append(result)
            
            #取出日志中的报警描述文本
            if(len(alarmLog[i])>=23):
                #print(i,alarmLog[i][23])
            #print(al['message'].split('||'))
                testDt_List.append(alarmLog[i][23])
            else:
                continue
            i+=1

        else:
            break;
            #print("no message!")
    return alarmLog, testDt_List

#使用算法处理从kafka中消费出来的数据
def kmeans_process(alarmLog,testDt_List):
    seg_Dt_List=Cluster.JieBaSplit(testDt_List)   #调用Cluster文件中的JieBaSplit()函数，将文本进行分词
    test_arr=Cluster.toTfidfVec(seg_Dt_List)      #调用toTfidfVec()函数，将分完词的文本转换成特征向量
    #dimension=66
    #test_arr_lowD=Cluster.DimenReduPCA(test_arr,dimension)
    k=10                                          #设置聚类簇数
    label=Cluster.kmeans(test_arr,k,alarmLog)     #调用kmeans()函数进行聚类
    label.sort(key=lambda x:x[35])                #将聚类结果安装分类标签排序
    return label
    #print(label)
    #result=pd.DataFrame(label)
    #result.to_csv('./data/kmeans2_Result.csv',index=False,header=None)

#将算法处理后的结果插入到数据库
def InsertDB(label):
    #创建数据连接，host为数据库服务所在的设备ip，port为整数形式的端口号，
    #user是登陆数据库的用户名，passwd数据库登陆密码，db要插入的数据库的名字，
    #charset设置插入数据库的编码，autocommit默认为False设置为True自动将执行sql结果提交
    db=pymysql.connect(host="80.2.238.228",
                       port=8886,
                       user="testuser",
                       passwd="ebank#1Dcc",
                       db="testdb",
                       charset="utf8",
                       autocommit=True)
    cursor=db.cursor()   #创建游标
    classid=int(label[0][35])  #获取每条记录的分类标签
    #计算每条记录的父事件的id
    SQL_SelectID='SELECT max(SyslogID) FROM poseidon_alarm_sys'
    cursor.execute(SQL_SelectID)  #先查询数据库中最后一条记录的SyslogID值
    id=cursor.fetchone()     #选取查询结果中的一条记录
    #如果没有记录，证明数据库为空，则开始父事件即为第一条记录的SyslogID，否则就是最后一条的SyslogID+1
    if(id[0]==None):   
        fatherEvent=1
    else:
        fatherEvent=id[0]+1
    #label[0].append(id)
    
    #逐条插入数据库
    for i in range(len(label)):
        #合并重复发生的报警记录
        #查询数据库表里是否已经有当前报警的记录，如果存在则将已有记录的发生时间更新成当前记录的发生时间，发生频率+1
        #否则将此条记录插入数据库，并将事件开始时间赋值为当前的发生时间
        SQL_SameRecord = "SELECT * FROM poseidon_alarm_sys WHERE NodeIP='%s'"%label[i][2]+" AND ComponentType='%s'"%label[i][6]+" AND SubComponent='%s'"%label[i][8]+" AND SummaryCN='%s'"%label[i][23]
        cursor.execute(SQL_SameRecord)
        recd = cursor.fetchone()
        #print(recd)
        Num = 1
        if (recd != None):
            if(recd[6]!=None):
                Num=int(recd[6])+1
            SQL_Update = "UPDATE poseidon_alarm_sys set Occurence='%s'"%label[i][34]+" , FREQUENCY=%d" %Num
            cursor.execute(SQL_Update)
        else:
            START_TIME = label[i][34]
            #print(START_TIME)
            #判断tally字段转为整数形式
            if(label[i][24]!=""):
                tally=int(label[i][24])
            else:
                tally=None
            
            label[i].append(fatherEvent) #插入字段父事件id（fatherEvent)

            sql="INSERT INTO poseidon_alarm_sys(FatherEvent,Class,START_TIME,Occurence,FREQUENCY,Customer,NodeAlias,NodeIP,BusinessName,AppName,AppShortName,ComponentType,Component,SubComponent,InstanceId,InstanceValue,EventName,EventNameCN,Type,EventType,CustomerSeverity,FirstOccurrence,LastOccurrence,SourceServerSerial,ACK_Time,Close_Time,OwnerGroup,Summary,SummaryCN,Tally,Site,OrgID,DevOrgID,ProcessMode,EnrichStatus,MaintainStatus,SMSFlag,Acknowledged,EventStatus) VALUES(%d,%d,'%s','%s',%d,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')"%(int(label[i][36]),int(label[i][35]),START_TIME,label[i][34],Num,label[i][0],label[i][1],label[i][2],label[i][3],label[i][4],label[i][5],label[i][6],label[i][7],label[i][8],label[i][9],label[i][10],label[i][11],label[i][12],label[i][13],label[i][14],label[i][15],label[i][16],label[i][17],label[i][18],label[i][19],label[i][20],label[i][21],label[i][22],label[i][23],tally,label[i][25],label[i][26],label[i][27],label[i][28],label[i][29],label[i][30],label[i][31],label[i][32],label[i][33])
            cursor.execute(sql)
            
            #如果分类标签换了即更改fatherEvent字段
            if(i<len(label)-1 and int(label[i+1][35])!=classid):
                classid=int(label[i+1][35])
                fatherEvent=cursor.lastrowid+1
    db.close()  #关闭数据库连接

if __name__ == '__main__':
    alarmLog, testDt_List=KafkaCon()
    if(len(testDt_List)>0):
        label=kmeans_process(alarmLog,testDt_List)
        InsertDB(label)