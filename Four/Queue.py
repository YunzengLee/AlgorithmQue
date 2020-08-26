#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宽度优先搜索"""
__author__ = "Liyunzeng"
import queue


class Soulution1():
    def flaten(self, root):
        # 每个点只进入一次出来一次，因此时间复杂度为O（n） 空间复杂度为最多的那一层q有多少点，最坏的情况就是O（n）
        res = []
        if root is None:
            return res
        q = queue.Queue()
        q.put(root)
        # res = []
        while not q.empty():
            size = q.qsize()
            level = []
            for i in range(size):  # #########去掉这个for循环，就变成了树的前序遍历
                node = q.get()
                level.append(node.val)
                if node.left is not None:
                    q.put(node.left)
                if node.right is not None:
                    q.put(node.right)
            res.append(level)

        return res

    def flatten2(self, root):
        # 上个函数去掉for循环 就变成了BFS前序遍历
        # 上面这句话说错了 不是去掉就变成前序遍历，去掉后每个节点的遍历顺序依然是与上题相同的
        # 分层遍历的目的只是为了记录一个当前层的结果 如上题的level
        res = []
        if root is None:
            return res
        q = queue.Queue()
        q.put(root)
        while not q.empty():
            node = q.get()
            res.append(node.val)
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)
        return res


class TreeNode():
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class GraphNode():
    def __init__(self, val=None, neighbor=None):
        self.val = val
        if neighbor == None:
            self.neighbor = []
        else:
            self.neighbor = neighbor


class Solution2():
    # 二叉树的序列化  做法是对二叉树进行分层记录
    def tree_serialization(self, root):
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

    # 反序列化
    def tree_re_serialization(self, string):
        if string == '':
            return None

        root = TreeNode()
        root.val = int(string[0])

        q = queue.Queue()
        q.put(root)

        length = len(string)
        idx = 1
        while not q.empty() and idx < length:
            node = q.get()
            node_new = TreeNode()
            if string[idx] != '#':  # 左节点安排上
                node_new.val = int(string[idx])
                node.left = node_new
                q.put(node_new)
            idx += 1

            node_new = TreeNode()  # 右节点安排上
            if string[idx] != '#':
                node_new.val = int(string[idx])
                node.right = node_new
                q.put(node_new)
            idx += 1
        return root

    def get_graph(self, n, edges):
        """

        :param n:
        :param edges:
        :return:graph
        """
        '''判图为树  图为树的条件：边的数量为n-1； 所有节点是联通的。
        1 首先将给出的信息表示出这个图 如果这个图中每个节点的值互不相同 就可以用dict表示
        2 从某个节点开始，遍历该图，看遍历的节点数是否等于给出的节点数。（判断联通性）
        '''
        '''
        Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
        write a function to check whether these edges make up a valid tree.
        For example:
        Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
        Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.
        '''
        dict = {}
        for i in range(n):
            dict[i] = set()
        for i in range(len(edges)):
            u = edges[i][0]
            v = edges[i][1]
            dict[u].add(v)
            dict[v].add(u)
        return dict

    def graph_is_tree(self, n, edges):
        '''判图为树'''
        '''class Node():
                 def __init__(self):
                    self.val=None
                    self.neighbor=[]  # 图节点可以这样表示，但该题用的是dict表示的
        '''
        if len(edges) != n - 1:
            return False
        graph = self.get_graph(n, edges)  # 首先用dictionary表示出该graph
        q = queue.Queue()
        s = set()
        q.put(0)
        s.add(0)
        while not q.empty():
            node = q.get()
            for i in graph[node]:
                if i not in s:
                    s.add(i)
                    q.put(i)
        if len(s) != n:
            return False
        else:
            return True

    def clone_graph(self, node):
        # 克隆图
        # ########该题的有意思之处在于用dict结构进行旧图节点与新图节点之间的映射关系。dict可用于做映射！！！！！！
        """

        :param node:
        :return:cloned graph
        """
        # get all nodes
        root = node
        # 实例是个可变对象 因此下面的循环里不能用node 也就是说不能改变node值
        q = queue.Queue()
        nodes = set()
        q.put(node)
        nodes.add(node)
        while not q.empty():
            n = q.get()
            for i in n.neighbor:
                if i not in nodes:
                    q.put(i)
                    nodes.add(i)
        # 此时nodes里面是所有找到的节点
        # 克隆值
        maping = {}
        for nod in nodes:
            maping[nod] = GraphNode(nod.val, [])
        # 克隆边
        for nod in nodes:
            newNode = maping[nod]
            for neighbor in nod.neighbor:
                newNode.neighbor.append(maping[neighbor])
        return maping[root]

    def searchGraphNode(self, node, target):
        """
        一个图，含有节点node，找出离node最近的value等于target的节点
        :param node:
        :param target:
        :return:
        """
        q = queue.Queue()
        s = set()
        q.put(node)
        while not q.empty():
            node = q.get()
            if node.val == target:
                return node
            for neighbor in node.neighbors:
                if neighbor not in s:
                    q.put(neighbor)
                    s.add(neighbor)
        return None

    def searchGraphNodeNew(self, node, target):
        """
        一个图，含有节点node，找出所有离node最近的value等于target的节点 与上一个不一样 要求找出所有最近的节点 不止一个节点
        :param node:
        :param target:
        :return:
        """
        res = []
        q = queue.Queue()
        s = set()
        q.put(node)
        while not q.empty():
            if res != []:
                break
            size = q.qsize()
            for i in range(size):
                node = q.get()
                if node.val == target:
                    res.append(node)
                for neighbor in node.neighbors:
                    if neighbor not in s:
                        q.put(neighbor)
                        s.add(neighbor)
        return res


class Solution3():
    # 拓扑排序
    def tuopu_sorting(self, graph):
        # 首先统计各节点的入度：
        indegree = {}
        for node in graph:
            indegree[node] = 0
        for node in graph:
            for neighbor in node.neighbors:
                indegree[neighbor] += 1

        # 找出入度为0的点放入结果表中
        res = []
        for key in indegree:
            if indegree[key] == 0:
                res.append(key)

        # BFS遍历
        q = queue.Queue()
        for i in res:
            q.put(i)

        while not q.empty():
            node = q.get()
            for neighbor in node.neighbors:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    q.put(neighbor)
                    res.append(neighbor)
        if len(res) == len(graph):
            return res
        else:
            return None


class Soulution4():
    def isIsland(self, matrix):
        """
        给一个0 1矩阵判断里面有几个岛屿，岛屿即由1联通的块
        :param matrix:
        :return:
        """
        island = 0
        posionx = [1, 0, -1, 0]
        posiony = [0, 1, 0, -1]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):  # 循环遍历每一个点
                if matrix[i][j] == 1:  # 如果找到一个点为1， island+1
                    island += 1
                    q = queue.Queue()
                    q.put((i, j))  # 将值为1的该点加入队列
                    while not q.empty():
                        posion = q.get()
                        matrix[posion[0]][posion[1]] = 0  # 从队列中取出点，并置0
                        for k in range(4):  # 向该点的四个方向找相邻位置的元素
                            new_posionx = i + posionx[k]
                            new_posiony = j + posiony[k]
                            if new_posionx < 0 or new_posionx > len(matrix) or new_posiony < 0 or new_posiony > len(
                                    # 如果超出边界则跳过
                                    matrix[0]):
                                continue
                            if matrix[new_posionx][new_posiony] == 1:  # 如果相邻位置元素是1，加入队列
                                q.put((new_posionx, new_posiony))

        return island

    def zombieInMatrix(self, matrix):
        """
        下一个函数的改进版，更完善
        :param matrix:
        :return:
        """
        PEOPLE = 0
        ZOMBIE = 1
        WALL = 2
        matrix_width = len(matrix)
        matrix_height = len(matrix[0])
        q = queue.Queue()
        s = set()

        people_num = 0

        for i in range(matrix_width):
            for j in range(matrix_height):
                if matrix[i][j] == PEOPLE:
                    people_num += 1
                if matrix[i][j] == ZOMBIE:
                    q.put((i, j))
                    s.add((i, j))
        if people_num == 0:
            return 0
        day = 0
        deltax = [0, 1, 0, -1]
        deltay = [1, 0, -1, 0]
        # zombied_people=0
        while not q.empty():
            day += 1
            size = q.qsize()
            for i in range(size):
                zombie_posion = q.get()
                for j in range(4):
                    new_posionx = zombie_posion[0] + deltax[j]
                    new_posiony = zombie_posion[1] + deltay[j]
                    if new_posionx < 0 or new_posiony < 0 or new_posionx >= m or new_posiony >= n:
                        continue
                    if matrix[new_posionx][new_posiony] == PEOPLE:
                        people_num -= 1
                        if people_num == 0:
                            return day
                        matrix[new_posionx][new_posiony] = 1
                        q.put((new_posionx, new_posiony))
        if people_num != 0:
            return -1

    def zombieinmatrix(self, matrix):
        """
        矩阵中，1表僵尸，0表人，2表墙。每一天 1将周围的0变为1，不能隔墙。问几天后0全变1？ 若无法全变1则返回-1
        :param matrix:
        :return:
        """
        day = 0
        deltax = [0, 1, 0, -1]
        deltay = [1, 0, -1, 0]
        q = queue.Queue()
        s = set()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1:
                    q.put((i, j))
                    s.add((i, j))
        while not q.empty():
            size = q.qsize()
            day += 1
            for k in range(size):
                zombie_p = q.get()
                for l in range(4):
                    new_posionx = zombie_p[0] + deltax[l]
                    new_posiony = zombie_p[1] + deltay[l]
                    if new_posionx < 0 or new_posiony < 0 or new_posionx >= len(matrix) or new_posiony >= len(
                            matrix[0]):
                        continue

                    if matrix[new_posionx][new_posiony] == 0 and (new_posionx, new_posiony) not in s:
                        matrix[new_posionx][new_posiony] = 1
                        q.put((new_posionx, new_posiony))
                        s.add((new_posionx, new_posiony))

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    # print(matrix)
                    return -1
        return day - 1

    def knight_short_path(self, matrix, source, target):
        """
        骑士按日字在0 1 的matrix中行走，1为障碍物，不可到达，问从source到target的最短路径需几步？
        :param matrix:
        :param source:
        :param target:
        :return:
        """
        if matrix == None:
            return False
        q = queue.Queue()
        s = set()
        q.put((source[0], source[1]))
        s.add((source[0], source[1]))
        step = 0
        deltax = [1, -1, 1, -1, 2, -2, 2, -2]
        deltay = [2, -2, -2, 2, 1, -1, -1, 1]
        while not q.empty():
            size = q.qsize()
            step += 1
            for i in range(size):
                posion = q.get()
                for j in range(8):
                    new_posionx = posion[0] + deltax[j]
                    new_posiony = posion[1] + deltay[j]
                    if new_posionx < 0 or new_posiony < 0 or new_posionx >= len(matrix) or new_posiony >= len(
                            matrix[0]):
                        continue
                    if new_posionx == target[0] and new_posiony == target[1]:
                        return step
                    if (new_posionx, new_posiony) not in s:
                        s.add((new_posionx, new_posiony))
                        q.put((new_posionx, new_posiony))


def pantuweishu(n, edges):
    # 判图为树 （重写）
    """

    :param n: jiediangeshu
    :param edges: bian d liebiao
    :return:
    """
    if len(edges) != n - 1:
        return False
    dic = {}
    for i in range(n):
        dic[i] = set()
    for i in range(len(edges)):
        u = edges[i][0]
        v = edges[i][1]
        dic[u].add(v)
        dic[v].add(u)
    q = queue.Queue()
    s = set()
    q.put(0)
    s.add(0)
    while not q.empty():
        node = q.get()
        for i in dic[node]:
            if i not in s:
                q.put(i)
                s.add(i)
    return len(s) == n


def kelongtu(node):
    # # 克隆图（重写）
    q = queue.Queue()
    s = set()
    q.put(node)
    s.add(node)
    while not q.empty():
        nod = q.get()
        for i in nod.neighbor:
            if i not in s:
                q.put(i)
                s.add(i)
    mapping = {}
    for i in s:
        mapping[i] = GraphNode(i.val, [])
    for i in s:
        for neighbor in i.neighbors:
            mapping[i].neighbor.append(mapping[neighbor])
    return mapping[node]


if __name__ == "__main__":
    a = Soulution4()
    matrix = [
        [0, 1, 2, 0, 2],
        [1, 0, 0, 2, 1],
        [0, 1, 0, 0, 0]
    ]
    day = a.zombieinmatrix(matrix)
    print(day)
