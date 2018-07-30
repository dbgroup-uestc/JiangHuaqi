import re
import math
from  collections import Counter
class POICounter():
    #计算POI出现频率,输入文件中记录必须如：USER_1083	LOC_1500	1.4447635942370927,103.76816511154175	06:57	0
    def __init__(self,poi_list):
        self.poi_list = poi_list
        self.count_res = None
        self.user_dict = None
        self.poi_dict = {}


        self.POI_Count(self.poi_list)

    def POI_Count(self,poi_list):
        count = 0
        filter_poi_list = []
        documents = []
        poi_dict = {}
        for line in poi_list:
            tempLine = re.split('\t|,', line.strip())
            tempLine[0] = tempLine[0][5:]
            tempLine[1] = tempLine[1][4:]
            documents.append(tempLine)
        #形成用户行程和poi字典，属性为{'pid':{"lat":1,"lon":1}
        USERS = {}
        class CheckIn:
            def __init__(self,pid,Lat,Lon,day,time):
                self.pid = int(pid)
                self.Lat = float(Lat)
                self.Lon = float(Lon)
                self.day = int(day)
                self.time = time
            def __repr__(self):
                return repr((self.pid,self.Lat,self.Lon,self.day,self.time))
        for record in documents:
            if record[0] not in USERS.keys():
                USERS[record[0]] = []
            USERS[record[0]].append(CheckIn(record[1],record[2],record[3],record[5],record[4]))

            self.poi_dict[int(record[1])] = {"lat":float(record[2]),"lon":float(record[3])}

        #排序用户行程
        for User in USERS:
            temp = USERS[User]
            temp=sorted(temp,key=lambda checkin:(checkin.day,checkin.time))
            USERS[User] = temp
        self.user_dict = USERS

        #将用户行程变成句子
        for User in USERS:
            sentence = ""
            for visit in USERS[User]:
                if visit != len(USERS[User]) - 1:
                    sentence += str(visit.pid) + " "
                else:
                    sentence += str(visit.pid)
            filter_poi_list.append(sentence)
        c = Counter()
        for sentence in filter_poi_list:
            templist = sentence.split(" ")
            c.update(sentence.split(" "))
        del c['']
        self.count_res = c



if __name__ == '__main__': #用于模块测试的技巧，当程序从主模块运行，则不会执行这个模块
    file_object = open("C:\\Users\\Client\\PycharmProjects\\JiangHuaqi\\code\\datasets\\poidata\\Foursquare\\train.txt","rU")
    pc = POICounter(file_object)
    print(pc.count_res)
