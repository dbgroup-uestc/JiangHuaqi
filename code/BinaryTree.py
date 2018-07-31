import numpy as np
from POICount import POICounter
import copy

def isContain(a,b,c):
    if a - b >= 0 and  c -a >= 0:
        return True
    else:
        return  False
class TreeNode():
    def __init__(self,region):
        self.possibility = 1 #保存POI路径概率
        self.left = None
        self.right = None
        self.value = "" #叶子节点存储单词本身，非叶子节点存储中间向量
        self.Huffman = "" #存储the Huffman code
        self.frequence = 0 #用于Huffman Tree的建立
        self.region =region
class Reigion():
    #a,b,c,d分别表示最左边的经度，最右边的经度，上方的纬度，下方的纬度
    def __init__(self,poi_list,line):
        self.poi_list = poi_list
        [self.a,self.b,self.c,self.d] = line

    #分隔区域，0按照经度划分（竖线），1按照纬度划分（横线）
    def Spilt(self,method=0):
        if method == 0:
            midlon = (self.a + self.b)/2
            left_set = {}
            right_set = {}
            for poi in self.poi_list:
                if self.poi_list[poi]["lon"] <= midlon:
                    left_set[poi] = self.poi_list[poi]
                else:
                    right_set[poi] = self.poi_list[poi]
            left_region = Reigion(left_set,[self.a,midlon,self.c,self.d])
            right_region = Reigion(right_set,[midlon,self.b,self.c,self.d])
            return [left_region,right_region]
        else:
            midlat = (self.c+self.d)/2
            above_set = {}
            below_set = {}
            for poi in self.poi_list:
                if self.poi_list[poi]["lat"] <= midlat:
                    above_set[poi] = self.poi_list[poi]
                else:
                    below_set[poi] = self.poi_list[poi]
            above_region = Reigion(above_set,[self.a,self.b,self.c,midlat])
            below_region = Reigion(below_set,[self.a,self.b,midlat,self.d])
            return [above_region,below_region]

    #判断两个区域是否交叉，返回交叉的面积占POI影响范围的比例，region1是POI影响区域，region2是节点区域
    def IsCross(region1,region2):
        if isContain(region1.a,region2.a,region2.b) and isContain(region1.c,region2.c,region2.d):
            return (region2.b - region1.a)*(region2.d - region1.c)/(0.1*0.1)
        elif isContain(region1.b,region2.a,region2.b) and isContain(region1.c,region2.c,region2.d):
            return (region1.b - region2.a)*(region2.d - region1.c)/(0.1*0.1)
        elif isContain(region1.b,region2.a,region2.b) and isContain(region1.d,region2.c,region2.d):
            return (region1.b - region2.a)*(region1.d - region2.c)/(0.1*0.1)
        elif isContain(region1.a,region2.a,region2.b) and isContain(region1.d,region2.c,region2.d):
            return (region2.b - region1.a)*(region1.d - region2.c)/(0.1*0.1)
        else:
            return 0.0

    def Add_Poi(self,poi):
        for poi_id in poi:
            self.poi_list[poi_id] = poi[poi_id]

class BinaryTree():
    def __init__(self,pc,vec_len=200):
        self.vec_len = vec_len
        self.root = None
        self.pc = pc
        #从poi_dict中寻找决定区域的四条经纬线
        min = 0
        max = 360
        [a,b,c,d] = [max,min,max,min]
        self.theta = 0.1
        for poi in pc.poi_dict:
            if pc.poi_dict[poi]["lon"] < a:
                a = pc.poi_dict[poi]["lon"]
            if pc.poi_dict[poi]["lon"] > b:
                b = pc.poi_dict[poi]["lon"]
            if pc.poi_dict[poi]["lat"] < c:
                c =pc.poi_dict[poi]["lat"]
            if pc.poi_dict[poi]["lat"] > d:
                d =pc.poi_dict[poi]["lat"]
        self.root = TreeNode(Reigion(pc.poi_dict,[a,b,c,d]))
        self.Built_Btree(self.root)
        self.InfuenceProcess()

    def Built_Btree(self, fnode):
        if fnode.region.b-fnode.region.a > 2*self.theta:
            [fnode.left, fnode.right] = self.Spilt(0,fnode)
            self.Built_Btree(fnode.left)
            self.Built_Btree(fnode.right)
        elif fnode.region.d - fnode.region.c > 2*self.theta:
            [fnode.left, fnode.right] = self.Spilt(1,fnode)
            self.Built_Btree(fnode.left)
            self.Built_Btree(fnode.right)
    def Spilt(self,method,fnode):
        [left_region,right_region] = fnode.region.Spilt(method)
        left_node = TreeNode(left_region)
        right_node = TreeNode(right_region)
        return [left_node,right_node]



#添加影响力
    def InfuenceProcess(self):
        #寻找所有的叶子节点
        leaves = []
        stack = []
        stack.append(self.root)
        while len(stack) != 0: #先序遍历
            node = stack.pop()
            if node.left == None and node.right == None:
                leaves.append(node)
            else:
                if node.right != None:
                    stack.append(node.right)
                if node.left != None:
                    stack.append(node.left)
        visited_list = []
        for leaf in leaves:
            temp_list = leaf.region.poi_list
            for poi in temp_list:
                if poi not in visited_list:
                    visited_list.append(poi)
                    lat = temp_list[poi]["lat"]
                    lon = temp_list[poi]["lon"]
                    influen_area = Reigion(None,[lon-self.theta/2,lon+self.theta/2,lat-self.theta/2,lat+self.theta/2])
                    possibility = 1
                    for other_leaf in leaves:
                        if other_leaf != leaf:
                            temp_possibility = Reigion.IsCross(influen_area,other_leaf.region)
                            if temp_possibility != 0:
                                temp_info = copy.copy(temp_list[poi])
                                temp_info["possibility"] = temp_possibility
                                new_poi = {poi:temp_info}
                                other_leaf.region.Add_Poi(new_poi)
                                possibility -= temp_possibility
                    temp_list[poi]["possibility"] = possibility
                    temp_list[poi]["possibility"] = possibility

#测试所有poi的概率为1
        for poi in visited_list:
            nums = 0
            possibility = 0
            for leaf in leaves:
                if poi in leaf.region.poi_list.keys():
                    possibility += leaf.region.poi_list[poi]["possibility"]
            if abs(possibility - 1) > 1e-10:
                print("bad")





if __name__== '__main__':
    file_object = open("C:\\Users\\Client\\PycharmProjects\\JiangHuaqi\\code\\datasets\\poidata\\Foursquare\\train.txt","rU")
    pc = POICounter(file_object)
    bt = BinaryTree(pc)
    print("hi")
