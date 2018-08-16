# -*- coding: utf-8 -*-
"""
Created on Apr 28 2016
Extracting vocabulary from Youdao dictionary
The vocabulary text file should be code as utf-8

<INPUT>
file_in: the exported vocabulary from Youdao
</INPUT>

<OUTPUT>
file_out: the file to save the English words. Default file name is
            new_words_'time'.txt ('time' is the local date)
<OUTPUT>

@author: sinit
"""
import codecs,time
file_in = r'D:\voc.txt'
outname = 'new_words'+'_'+time.strftime("%Y-%m-%d",time.localtime())+".txt"
file_out = r'D:\\'+outname
fs = codecs.open(file_in, 'r','utf-8')
vocabulary = fs.readlines()
fs.close()
word = []
word.append(vocabulary[0].split()[1])
def is_chinese(uchar):
#Judge if a unicode is Chinese
    if (uchar >=u'/u4e00')&(uchar<=u'/u9fa5'):
        return True
    else:
        return False
def is_zh (c):
        x = ord (c)
        # Punct & Radicals
        if x >= 0x2e80 and x <= 0x33ff:
                return True

        # Fullwidth Latin Characters
        elif x >= 0xff00 and x <= 0xffef:
                return True

        # CJK Unified Ideographs &
        # CJK Unified Ideographs Extension A
        elif x >= 0x4e00 and x <= 0x9fbb:
                return True
        # CJK Compatibility Ideographs
        elif x >= 0xf900 and x <= 0xfad9:
                return True

        # CJK Unified Ideographs Extension B
        elif x >= 0x20000 and x <= 0x2a6d6:
                return True

        # CJK Compatibility Supplement
        elif x >= 0x2f800 and x <= 0x2fa1d:
                return True

        else:
                return False
for i in range(1,len(vocabulary)):
    line = vocabulary[i].split()
    if vocabulary[i].split()[0][:-1].isdigit():
        newword = vocabulary[i].split()[1]
        if is_zh(newword[0]):
            continue
        else:
            word.append(vocabulary[i].split()[1])
fs = open(file_out, 'w+')
for line in word:
    fs.write(line)
    fs.write('\n')
fs.close()

print('Assignment Done!')

