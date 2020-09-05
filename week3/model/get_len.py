import numpy as np
import os
# import  matplotlib.pyplot as plt

ppgs_file = os.listdir('/home/team06/week3_code/db1/ppgs')

all_lenth = {}
cnt = 0
print(len(ppgs_file))
for i in range(len(ppgs_file)):
    p = np.load('/home/team06/week3_code/db1/ppgs/'+ppgs_file[i])
    lenth = p.shape[0]
#     print(lenth)
    cnt+=lenth
    if lenth in all_lenth:
        all_lenth[str(lenth)]+=1
    else:
        all_lenth[str(lenth)] = 1

# print(all_lenth)
d = sorted(all_lenth.items(),key = lambda all_lenth:all_lenth[0])
# print(d)
print(d['112'])
# print(cnt*1.0 /len(ppgs_file) )
# # print(all_lenth)
# mid = {}
# cnt = 0
# for i, key in enumerate(d):#Circulate both index and value(Here is key)
#     cnt = int(i)+cnt
#     mid[key] = cnt
    
# print(mid)