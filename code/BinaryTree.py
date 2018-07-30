import numpy as np
from POICount import POICounter
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
            above_set = Reigion(above_set,[self.a,self.b,self.c,midlat])
            below_set = Reigion(above_set,[self.a,self.b,midlat,self.d])
            return [above_set,below_set]

class BinaryTree():
    def __init__(self,pc,vec_len=200):
        self.vec_len = vec_len
        self.root = None
        self.pc = pc
        #从poi_dict中寻找决定区域的四条经纬线
        min = 0
        max = 360
        [a,b,c,d] = [max,min,max,min]
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

    def Built_Btree(self, fnode):
        if fnode.region.b-fnode.region.a > 0.2:
            [fnode.left, fnode.right] = self.Spilt(0,fnode)
            self.Built_Btree(fnode.left)
            self.Built_Btree(fnode.right)
        elif fnode.region.d - fnode.region.c > 0.2:
            [fnode.left, fnode.right] = self.Spilt(1,fnode)
            self.Built_Btree(fnode.left)
            self.Built_Btree(fnode.right)

    def Spilt(self,method,fnode):
        [left_region,right_region] = fnode.region.Spilt(method)
        left_node = TreeNode(left_region)
        right_node = TreeNode(right_region)
        return [left_node,right_node]




if __name__== '__main__':
    file_object = open("C:\\Users\\Client\\PycharmProjects\\JiangHuaqi\\code\\datasets\\poidata\\Foursquare\\train.txt","rU")
    pc = POICounter(file_object)
    bt = BinaryTree(pc)
    print("hi")
