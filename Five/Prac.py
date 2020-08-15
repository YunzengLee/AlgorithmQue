def combination_sum(candidate,target):
    def helper(candidate,startidx,cur_sum,subset,target,res):
        print(subset,cur_sum)
        if cur_sum>target:
            return
        if cur_sum==target:
            res.append(list(subset))
            return
        for i in range(startidx,len(candidate)):
            if i!=0 and candidate[i]==candidate[i-1]:
                continue
            # cur_sum+=candidate[i]    # 这一句绝对不行，一定要在下面的helper函数调用时再加上当前值
            # 若在此处更改cur_sum值，下一次循环时cur_sum值就已经改变了
            subset.append(candidate[i])
            helper(candidate,i,cur_sum+candidate[i],subset,target,res)
            subset.pop()




    res=[]
    if candidate is None or candidate==[]:
        return res
    startidx=0
    subset=[]
    helper(candidate,startidx,0,subset,target,res)
    return res
def huiwen(string):
    def helper(startidx,string,subs,res):
        print(subs)
        if startidx==len(string):
            res.append(list(subs))
            return
        for i in range(startidx+1,len(string)+1):
            substr=string[startidx:i]
            if substr==substr[::-1]:
                subs.append(substr)
                helper(i,string,subs,res)
                subs.pop()


    res=[]
    if string is None or string=='':
        return res
    startidx=0
    subs=[]
    helper(startidx,string,subs,res)
    return res
def permutations(l):
    def helper(l,subl,res):
        if len(subl)==len(l):
            res.append(list(subl))
        for i in range(len(l)):
            if l[i] not in subl:
                subl.append(l[i])
                helper(l,subl,res)
                subl.pop()


    l.sort()
    res=[]
    helper(l,[],res)
    return res
def permutationsii(l):
    def helper(l,sub_s,visited,res):
        if len(sub_s)==len(l):
            res.append(list(sub_s))
        for i in range(len(l)):
            if i != 0 and l[i]==l[i-1] and visited[i-1]==0:
                continue
            if visited[i]==0:
                visited[i]=1
                sub_s.append(l[i])
                helper(l,sub_s,visited,res)
                sub_s.pop()
                visited[i]=0


    l.sort()
    visited=[0 for i in range(len(l))]
    res=[]
    sub_s=[]
    helper(l,sub_s,visited,res)
    return res
import queue
def worfdladderii(dict,source,target):
    #
    # 所谓的dfs和bfs结合并不是同时使用 而是先用bfs找出所有节点与终点的距离
    # 再用dfs从起点向距离减小的方向搜索  在子路径添加新节点的过程中，首先要判断要加入的节点是否离当前节点到终点的距离又进了一步
    # 若近了一步 则加入路径中 否则跳过，这样就保证了搜索得到的路径一定是最短路径，不会出现绕远的情况
    # 综上 bfs找出所有节点与终点的距离，这些距离信息在dfs时用来筛选绕远路的节点，从而保证得到的所有路径都是最短路径
    res=[]
    q=queue.Queue()
    s=set()
    subset=[source]
    q.put(source)
    s.add(source)
    # while not q.empty():
    #     size=q.qsize()
    #     for i in range(size):
    #         pass
    #

    def dfs(subset,cur,target,res):
        if cur==target:
            res.append(list(subset))
            return
        for nextnode in dict[cur]:
            if nextnode not in s:
                s.add(nextnode)
                subset.append(nextnode)
                # cur=nextnode  # 这个变量最好不要改 直接在dfs的函数调用中 让传递参数写成nextnide
                # 这一层的cur变量不应该指向下一层的节点
                dfs(subset,nextnode,target,res)
                subset.pop()
        # 照这个写法 最后res里应该包含从起点到终点的所有路径，保留最短路径还是要bfs 怎么穿插进去呢？
    def bfs(curnode,subset,target,res):
        if res!=[]:
            return
        while not q.empty():
            size=q.qsize()
            for i in range(size):
                 pass
        if curnode==target:
            res.append(list(subset))

        for nextnode in dict[curnode]:
            if nextnode not in s:
                q.put(nextnode)
                s.add(nextnode)



if __name__=='__main__':
    candidate = [3, 2, 2,3]
    # print(combination_sum(candidate,5))
    string='ababc'
    # print(huiwen(string))
    print(permutationsii(candidate))
'''
1,1,2,2

         1                        2   
  1,1        1,2              2,1           2,2  
1,1,2    1,2,1  1,2,2     2,1,1  2,1,2     2,2,1
1,1,2,2 1,2,1,2  1,2,2,1  2,1,1,2 2,1,2,1  2,2,1,1
'''