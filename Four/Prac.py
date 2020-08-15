import queue
def flatten(root):

    res = []
    if root is None:
        return res
    q=queue.Queue()
    # s=set()
    q.put(root)
    # s.add(root)
    while not q.empty():
        size=q.qsize()
        level=[]
        for i in range(size):
            node=q.get()
            level.append(node.val)
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)
        res.append(level)
    return res
class TreeNode():
    def __init__(self,val=None,right=None,left=None):
        self.val=val
        self.right=right
        self.left=left
def serialization(root):
    string=''
    if root is None:
        return string
    q=queue.Queue()
    q.put(root)
    while not q.empty():
        size=q.qsize()
        for i in range(size):
            node=q.get()
            if node is not None:
                string+=str(node.val)
                q.put(node.left)
                q.put(node.right)
            else:
                string+='#'
    return string
def re_serilization(string):
    if string=='':
        return None
    length=len(string)
    root=TreeNode(val=string[0])
    q=queue.Queue()
    q.put(root)
    idx=1
    while idx<length:
        node=q.get()
        if string[idx]!='#':
            new_node=TreeNode(val=int(string[idx]))
            q.put(new_node)
            node.left=new_node
        idx+=1
        if string[idx]!='#':
            new_node=TreeNode(val=int(string[idx]))
            q.put(new_node)
            node.right=new_node
        idx+=1
    return root


def tree_serialization(root):
    res = ''
    if root is None:
        return res
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        size = q.qsize()
        for i in range(size):
            node = q.get()
            res += str(node.val)
            if node.left is not None:
                q.put(node.left)
            else:
                res += '#'
            # pass
            if node.right is not None:
                q.put(node.right)
            else:
                res += '#'
        # pass
    return res
def pantuweishu(n,edges):

    dicto={}
    for i in range(n):
        dicto[i]=set()
    for i in edges:
        dicto[i[0]].add(i[1])
        dicto[i[1]].add(i[0])

    if len(edges)!=n-1:
        return False

    q=queue.Queue()
    s=set()
    q.put(0)
    s.add(0)
    while not q.empty():
        node=q.get()
        for i in dicto[node]:
            if i not in s:
                s.add(i)
                q.put(i)
    if len(s)!=n:
        return False
    return True
def clone_graph(node):
    dict={}
    q=queue.Queue()
    s=set()
    root=node
    q.put(root)
    s.add(root)
    while not q.empty():
        node=q.get()
        dict[node]=None
        for sub_node in node.neighbors:
            if sub_node not in s:
                q.put(sub_node)
                s.add(sub_node)
    # 以上建立新旧节点之间的映射

    for node in s:
        for sub_node in node.neighbors:
            dict[node].neighbors.append(dict[sub_node])

    # 下面写的繁琐了 s里面已经有所有节点了 不需要再遍历一遍了
    q.put(root)
    s=set()
    s.add(root)
    while not q.empty():
        node=q.get()
        for sub_node in node.neighbors:
            dict[node].neighbors.append(dict[sub_node])
            if sub_node not in s:
                s.add(sub_node)
                q.put(sub_node)

    return dict[root]
def tuopu_sort(graph):
    # 将graph中的节点拓扑排序
    indegree={}
    for node in graph:
        indegree[node]=0
    for node in graph:
        for sub_node in node.neighbors:
            indegree[sub_node]+=1
    q=queue.Queue()
    res=[]
    for node in indegree:
        if indegree[node]==0:
           res.append(node)
           q.put(node)
    while not q.empty():
        node=q.get()
        for sub_node in node.neighbors:
            indegree[sub_node]-=1
            if indegree[sub_node]==0:
                q.put(sub_node)
                res.append(sub_node)
    return res
def island_num(matrix):
    def isout(matrix,x,y):
        if x<0 or y<0 or x>=len(matrix) or y>=len(matrix[0]):
            return True
        return False
    island=0
    dex=[0,0,1,-1]
    dey=[1,-1,0,0]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(i,j)
            if matrix[i][j]==1:
                island+=1
                q=queue.Queue()
                q.put((i,j))
                while not q.empty():
                    position=q.get()
                    matrix[position[0]][position[1]]=0
                    for k in range(4):
                        new_x=position[0]+dex[k]
                        new_y=position[1]+dey[k]
                        if isout(matrix,new_x,new_y):
                            continue
                        if matrix[new_x][new_y]==1:
                            q.put((new_x,new_y))
    return island


def zombie(matrix):
    def out(matrix,x,y):
        if x<0 or y<0 or x>=len(matrix) or y>=len(matrix[0]):
            return True
        return False
    PEOPLE=0
    ZOMBIE=1
    WALL=2
    dex=[0,0,1,-1]
    dey=[1,-1,0,0]
    days=0
    peoplenum=0
    q=queue.Queue()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]==1:
                q.put((i,j))
            if matrix[i][j]==0:
               peoplenum+=1
    if peoplenum==0:
        return days

    while not q.empty():
        size=q.qsize()
        for i in range(size):
            days+=1
            zombie_p=q.get()
            for k in range(4):
                newx=zombie_p[0]+dex[k]
                newy=zombie_p[1]+dey[k]
                if out(matrix,newx,newy):
                    continue
                if matrix[newx][newy]==0:
                    matrix[newx][newy]=1
                    q.put((newx,newy))
                    peoplenum-=1
                    if peoplenum==0:
                        return days

    return -1

def knight_short_path(matrix,source,target):
    steps=0
    if source==target:
        return steps
    dex=[]
    dey=[]

    q=queue.Queue()
    s=set()
    q.put(source)
    s.add(source)
    while not q.empty():
        steps+=1
        size=q.qsize()
        for i in range(size):
            pos=q.get()
            for k in range(8):
                newx=pos[0]+dex[k]
                newy=pos[1]+dey[k]
                if out:
                    continue
                if (newx,newy)==target:
                    return steps
                if (newx,newy) not in s:
                    s.add((newx,newy))
                    q.put((newx,newy))

if __name__=='__main__':
    root=TreeNode(val=0)
    root.right=TreeNode(val=2)
    root.left=TreeNode(val=1)
    root.left.left=TreeNode(val=3)
    root.left.right=TreeNode(val=4)
    # root.right.left = TreeNode(val=5)
    root.right.right = TreeNode(val=6)
    # print(tree_serialization(root))

    a=[
        [0,1,1,0],
        [1,0,0,0],
        [1,0,0,1]
    ]
    print(island_num(a))

