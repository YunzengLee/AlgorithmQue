# def next_pos(x,y,matrix):
#     dex = [0,0,1,-1]
#     dey = [1,-1,0,0]
#     res = []
#     for i in range(4):
#         newx = x+dex[i]
#         newy = y+dey[i]
#         if newx<0 or newy<0 or newx>=len(matrix) or newy>=len(matrix[0]):
#             continue
#         res.append((newx,newy))
#     return res
# def dfs(matrix,x,y,word,s,startidx):
#     res = False
#     if startidx>=len(word):
#         return True
#     for next_position in next_pos(x,y,matrix):
#         if matrix[next_position[0]][next_position[1]] == word[startidx] and next_position not in s:
#             s.append(next_position)
#             res = dfs(matrix,next_position[0],next_position[1],word,s,startidx+1)
#             if res:
#                 return True
#             s.pop()
#     return res
# def ifExist(matrix,word):
#     r = len(matrix)
#     l = len(matrix[0])
#     for i in range(r):
#         for j in range(l):
#             if matrix[i][j] == word[0]:
#                 s = [(i,j)]
#                 res = dfs(matrix,i,j,word,s,1)
#                 if res:
#                     return res
#     return False
# matrix=[]
# while True:
#     s= input()
#     if not s:
#         break
#     matrix.append(s.split(' '))
# word = matrix.pop()[0]
# print(ifExist(matrix,word))
def tes():
    N = int(input())
    if N <7:
        print(N)
        return
    arr = [float('inf') for i in range(N)]
    arr[0]=1
    p2 = 0
    p3 = 0
    p5 = 0
    num2 = arr[p2]*2
    num3 = arr[p3]*3
    num5 = arr[p5]*5
    i = 1
    while i < N:
        minval = min(num2,num3,num5)

        if minval == arr[i-1]:
            continue
        else:
            arr[i] = minval
            i+=1
        if minval == num2:
            p2 += 1
            num2 = arr[p2]*2
        if minval == num3:
            p3+=1

            num3 = arr[p3]*3
        if minval == num5:
            p5 += 1

            num5 = arr[p5]*5

    print(arr[N-1])
tes()
if __name__ == '__main__':

    pass