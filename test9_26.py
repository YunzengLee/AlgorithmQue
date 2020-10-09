def dfs(curidx,arr,stack,subset,res):
    if curidx>=len(arr) and not stack:
        # print(subset)
        res.append(list(subset))
    if curidx<len(arr):
        # print('1')
        stack.append(arr[curidx])
        dfs(curidx+1,arr,stack,subset,res)
        stack.pop()

    if stack:
        # print('2')
        a=stack.pop()
        subset.append(a)
        dfs(curidx,arr,stack,subset,res)
        stack.append(a)
        subset.pop()

inparr = input().split(",")
res = []
dfs(0,inparr,[],[],res)
print(res)


# def dfs(curidx,arr,stack,subset,res):
#     if curidx>=len(arr) and not stack:
#         # print(subset)
#         res.append(list(subset))
#         return
#     if curidx<len(arr):
#         # print('1')
#         # stack.append(arr[curidx])
#         dfs(curidx+1,arr,stack+[arr[curidx]],subset,res)
#
#     if stack:
#         # print('2')
#         a=stack.pop()
#         # subset.append(a)
#         dfs(curidx,arr,stack,subset+[a],res)
#
# inparr = input().split(",")
# res = []
# dfs(0,inparr,[],[],res)
# print(len(res))
# print(res)