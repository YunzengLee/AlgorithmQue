# def dfs(startidx, hashMap, leftstep, subset, res):
#     if leftstep <= 0:
#         s = set(subset)
#         res[0] = max(res[0], len(s))
#         return
#     for next in hashMap[startidx]:
#         subset.append(next)
#         dfs(next, hashMap, leftstep - 1, subset, res)
#         subset.pop()
#
#
# nAndK = input().split(' ')
# n = int(nAndK[0])
# k = int(nAndK[1])
# arr = input().split(' ')
# hashMap = {}
# for i in range(n):
#     hashMap[i] = []
# arr = list(map(int,arr))
# for i in range(n-1):
#     hashMap[i+1].append(arr[i])
#     hashMap[arr[i]].append(i+1)
# res = [0]
# leftstep = k
# dfs(0,hashMap,leftstep,[0],res)
# print(res[0])


#
# boy_id = input().split(' ')
# girl_id = input().split(' ')
# tuple_num = int(input())
# parts = []
# for i in range(tuple_num):
#     part = input().split(' ')
#     parts.append(part)
# subset = []
# res= [0]
# def dfs(subset, startidx, parts,res):
#     print(subset)
#     if startidx >= len(parts):
#         print('res count')
#         print(subset)
#         part_num = len(subset)//2
#         res[0] = max(res[0],part_num)
#         return
#     for idx in range(startidx, len(parts)):
#         part = parts[idx]
#         if part[0] not in subset and part[1] not in subset:
#             subset.append(part[0])
#             subset.append(part[1])
#             dfs(subset,idx+1,parts,res)
#             subset.pop()
#             subset.pop()
# dfs(subset,0,parts,res)
# print(res[0])



string = input()
record = {}
for i in 'abcxyz':
    record[i] = 0

length = len(string)
for i in string:
    if i in record:
        record[i]+=1
def ifvaild(record):
    sign = True
    for i in record.values():
        if i %2 !=0:
            sign = False
            break
    return sign
