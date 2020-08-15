# def dfs(idx,cur_sum,string,target):
#     global res
#     # print(idx,cur_sum)
#     if idx>=len(string):
#         # print(idx,cur_sum)
#         if cur_sum==target:
#             res += 1
#         return
#     char = int(string[idx])
#     dfs(idx+1,cur_sum+(char),string,target)
#     dfs(idx+1,cur_sum-(char),string,target)
# t = int(input())
# ans=[]
# for i in range(t):
#     tuple_t = input().split()
#     print(tuple_t)
#     tuple_t[1] = int(tuple_t[1])
#     string=tuple_t[0]
#     target=tuple_t[1]
#     res=[0]
#     dfs(1,int(string[0]),string,target,res)
#     ans.append(res[0])
# for i in ans:
#     print(i)

# s='12345'
# res=0
# dfs(0,0,s,3)
# print(res)






