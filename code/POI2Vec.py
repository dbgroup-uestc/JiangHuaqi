import logging
import  re
import  functools
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

#读取数据
file_object = open("C:\\Users\\Client\\PycharmProjects\\JiangHuaqi\\code\\datasets\\poidata\\Foursquare\\train.txt","rU")
documents = []
try:
    for line in file_object:
        tempLine = re.split('\t|,',line.strip())
        tempLine[0]=tempLine[0][5:]
        tempLine[1]=tempLine[1][4:]
        documents.append(tempLine)
finally:
    file_object.close()
#形成每个用户的行程
USERS = {}
class CheckIn:
    def __init__(self,pid,Lat,Lon,day,time):
        self.pid = int(pid)
        self.Lat = Lat
        self.Lon = Lon
        self.day = int(day)
        self.time = time
    def __repr__(self):
        return repr((self.pid,self.Lat,self.Lon,self.day,self.time))
for record in documents:
    if record[0] not in USERS.keys():
        USERS[record[0]] = []
    USERS[record[0]].append(CheckIn(record[1],record[2],record[3],record[5],record[4]))
#排序用户行程
for User in USERS:
    temp = USERS[User]
    temp=sorted(temp,key=lambda checkin:(checkin.day,checkin.time))
    USERS[User] = temp
print("HI")
#寻找语句
'''Sentence = []
for user in USERS:
    User = USERS[user]
    for c in range(len(User)):
        cword = User[c]
        day = cword.day
        [hour,second] = cword.time.split(":")
        ctime = (int(day))*24+int(hour)+(int(second))/60
        sentence = []
        forward = 1
        backward = 1
        sentence.append(cword.pid)
        for i in range(len(User)):
            if forward:
                if c+i+1 > (len(User))-1:
                    forward = 0
                else:
                    oword = User[c+i+1]
                    day1 = oword.day
                    [hour1,second1] = oword.time.split(":")
                    otime = (int(day1))*24+int(hour1)+(int(second1))/60
                    if abs(ctime-otime) > 6:
                        forward = 0
                    else:
                        sentence.append(oword.pid)
            if backward:
                if c-i-1 < 0:
                    backward = 0
                else:
                    oword = User[c-i-1]
                    day1 = oword.day
                    [hour1,second1] = oword.time.split(":")
                    otime = (int(day1))*24+int(hour1)+(int(second1))/60
                    if abs(ctime-otime) > 6:
                        backward = 0
                    else:
                        sentence.insert(0,oword.pid)
            if backward==0 and  forward==0:
                if len(sentence) > 1:
                    if sentence not in Sentence:
                        Sentence.append(sentence)
                break

'''
for User in USERS:
    sentence = ""
    for visit in USERS[User]:
        if visit != len(USERS[User])-1:
            sentence += str(visit.pid) + " "
        else:
            sentence += str(visit.pid)
print("Hi")