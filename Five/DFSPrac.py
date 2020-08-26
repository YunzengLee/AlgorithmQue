class Solution1():
    def combination_sum(self, candidate, target):
        """
        从数组candidate中找出所有组合，每个组合的和等于target
        :param condidate: 不下降数组，可能有重复 即[1,2,2,3,5,7]
        :param target:
        :return:
        """
        sub_set = []
        res = []
        if candidate == None:
            return res
        candidate.sort()
        self.helper(candidate, 0, sub_set, 0, target, res)
        return res

    def helper(self, candidate, startidx, subset, cur_sum, target, res):
        if cur_sum == target:
            # print(cur_sum)
            # print(subset)
            res.append(list(subset))  # 注意此处放入的不能是可变对象 也就是说 不能直接放入subset 否则会改变

            return
        if cur_sum > target:  # 当当前和大于target时，就返回 这样就不会在进入下面的选数进subset的步骤
            return
        for i in range(startidx, len(candidate)):
            if i != startidx and candidate[i] == candidate[i - 1]:  # 去重机制 跳过列表中的重复元素
                # print('continue one time')
                # print(candidate[i])
                # print(candidate[i-1])
                continue
            """以下三句是DFS常用模板：  1.子集加元素2.进入下一层嵌套3.去掉加入的元素 """
            subset.append(candidate[i])

            # if subset==[1,1]:
            # print('start')
            self.helper(candidate, i, subset, cur_sum + candidate[i], target, res)
            # 该句中startidx从i开始，表示列表中的数字可以重复使用，若不能重复使用，则改为i+1
            # 若不允许重复使用，则改为i+1，但这样可能会超出列表下标范围 所以循环取数的区间要改为range（startidx，len（candidate）-1）
            # if subset==[1,1]:
            # print('end')

            # print(subset)

            subset.pop()

            # subset=subset[:-1]   # #用这个函数会出奇怪的bug，且找不到原因。很神奇  还是用pop吧
            # ###找到原因了 s=s[:2] 并不是把s指向的列表缩减了 而是重新定义了一个列表再让s指向这个新列表
            # print(subset)

            # 这个题看程序不好理解是如何找各种组合的，但画一个二叉树就能看懂了
            '''
            例 【1，2,2,3,4】
                   [1]                  [2]         [3]      [4]
            [1 2] [1 3] [1 4]      [2 3]  [2 4]    [3 4]
       [123]  [124]  [134]  
            '''
class Solution2():
    def palindrome_patitioning(self, string):
        """
        切分字符串string，使切分得到的所有子串都是回文串
        :param string:'aab'
        :return: [ ['aa','b'],['a','a','b']  ]
        """
        #  字符串的切分方案实际上是各个字符之间的间隔是否取入组合。 故共有2^(n-1)个组合，n为字符个数
        jiange = list(range(1, len(string)))
        subset = []
        Res = []
        if string == None:
            return Res
        if string == string[::-1]:
            Res.append([string])
        startidx = 0
        self.helper(string, startidx, subset, jiange, Res)
        return Res

    def helper(self, string, startidx, subset, jiange, Res):
        # 好像不用着急判断是否符合条件 这个函数应该先找出所有切分组合来；不对 有的切分出来已经有非回文串了，就应该停止深层遍历 所以还是要加入判断
        # 处理
        string_copy = string
        if subset != []:
            res = []
            '''# string_copy = string
            for i in subset:
                sub_str = string[:i]
                string = string[i:]
                res.append(sub_str)
            res.append(string)'''

            # 按照subset切分字符串
            substr = string[:subset[0]]
            res.append(substr)
            if len(subset) > 1:
                for i in range(len(subset) - 1):
                    substr = string[subset[i]:subset[i + 1]]
                    res.append(substr)
            res.append(string[subset[-1]:])

            # if subset==[1]:
            #     print(res)
            # if subset==[1,2]:
            #     print(res)
            substr = res[-2]
            if substr != substr[::-1]:  # 如果倒数第二个已经不是回文串 就不必再递归
                return

            is_huiwen = True
            for substr in res:
                if substr != substr[::-1]:
                    is_huiwen = False
                    break
            if is_huiwen:
                # Res.append(list(subset))
                Res.append(list(res))

            if subset[-1] == jiange[-1]:  # 如果已经无法再进行切分 则返回 不再进入下面的递归
                return

            # 下一层
        for i in range(startidx, len(jiange)):
            subset.append(jiange[i])
            # print(subset)
            self.helper(string_copy, i + 1, subset, jiange, Res)
            subset.pop()

    def partition(self, string):
        #  上一题的更方便的解法
        #  子集不再是间隔的取法，而是得到的子串
        results = []
        if string == None or len(string) == 0:
            return results
        partition = []
        self.partition_helper(string, 0, partition, results)
        return results

    def partition_helper(self, string, startidx, partition, results):
        if startidx == len(string):
            results.append(list(partition))
            return
        for i in range(startidx, len(string)):
            sub_string = string[startidx, i + 1]
            if sub_string != sub_string[::-1]:
                continue
            partition.append(sub_string)
            self.helper(string, i + 1, partition, results)
            partition.pop()





class Solution3():
    # 给一个不重复列表，找出所有排列
    def permutationsI(self, list_):
        result = []
        if list_ == None or list_ == []:
            return result
        if len(list_) == 1:
            return [list_]
        subset = []
        self.helper(subset, list_, result)
        return result

    def helper(self, subset, list_, result):
        # 出口
        if len(subset) == len(list_):
            result.append(list(subset))
            return
        for i in list_:
            if i in subset:
                continue
            subset.append(i)
            self.helper(subset, list_, result)
            subset.pop()


class Solution4():
    # 给一个包含重复元素的列表，找出所有排列
    # 难道要用集合来去重？  不是 ，用Solution6中的方法去重即可
    def permutationsII(self, list_):
        result = []
        if list_ == None or list_ == []:
            return result
        if len(list_) == 1:
            return [list_]
        subset = []
        list_.sort()
        visited = [0 for i in range(len(list_))]
        # #由于不能通过 看某元素是否已经存在于子集中 来判断该元素是否已被使用，所以要设置这样一个列表visited 来表示list_中的元素是否已被使用
        self.helper(subset, list_, visited, result)
        return result

    def helper(self, subset, list_, visited, result):
        # 出口
        if len(subset) == len(list_):
            result.append(list(subset))
            return
        for i in range(len(list_)):
            if i != 0 and list_[i] == list_[i - 1] and visited[i - 1] == 0:  # ####这个visited[i-1]==0的条件需要仔细体会
                continue
            if visited[i] == 1:
                continue
            subset.append(list_[i])
            visited[i] = 1
            self.helper(subset, list_, result)
            visited[i] = 0
            subset.pop()


class Solution5():
    # Q-queens问题
    '''将n个queen放在n*n的矩阵里，每行每列只能有一个queen，且任意两个的queen不能在对角位置上 问有几种放法（也就是把n个queen放在n*n的棋盘里且相互无法攻击）'''
    ''''''

    def q_queens(self, n):
        if n == 1:
            return [1]
        list_ = list(range(1, n + 1))
        result = []
        subset = []
        self.helper(list_, subset, result)
        return result

    def helper(self, list_, subset, result):
        if len(subset) == len(list_):
            result.append(list(subset))
            return
        for i in list_:
            if self.is_valid(subset, i):
                continue
            subset.append(i)
            self.helper(list_, subset, result)
            subset.pop()

    def is_valid(self, subset, i):
        if i in subset:
            return False
        # x新放入的点 行数为len（subset），列数为i
        row = len(subset)
        for k in range(subset):
            if k - row == subset[k] - i or k + subset[k] == row + i:
                return False
        return True


class Solution7():
    # NQueens问题的原答案
    def q_queens(self, n):
        if n == 0 or n == None:
            return None
        results = []
        cols = []
        self.search(results, cols, n)
        return results

    def search(self, results, cols, n):
        if len(cols) == n:
            results.append(self.draw_chess(cols))
            return
        for i in range(n):
            if not self.is_valid(cols, i):
                continue
            cols.append(i)
            self.search(results, cols, n)
            cols.pop()

    def draw_chess(self, cols):
        chessboard = []
        for i in range(len(cols)):
            sb = ''
            for j in range(len(cols)):
                sb += 'Q' if j == cols[i] else '.'
            chessboard.append(sb)
        return chessboard

    def is_valid(self, cols, column):
        # 判断column能否加入cols里面去
        row = len(cols) # row是
        for rowidx in range(len(cols)):
            if cols[rowidx] == column:
                # 如果cols里已经有column则不行
                return False
            # 下面两个if是判断两个对角线上是否有其他皇后

            if rowidx + cols[rowidx] == row + column:
                return False
            if rowidx - cols[rowidx] == row - column:
                return False
        return True


class Solution6():
    def subset(self, num):
        # 给一个可能含有重复数字的数组，给出所有子集
        # ###########该题是一道经典题，里面用到的去重思路，在任何组合或排列问题中都可套用。
        '''
        例： 【1，2，2】
        return [
        [],[1],[2],[1,2],[1,2,2]
        ]
        '''
        result = []
        if num == None or len(num) == 0:
            return result
        startidx = 0
        subset = []
        num.sort()
        self.helper(subset, startidx, num, result)
        return result

    def helper(self, subset, startidx, num, result):
        result.append(list(subset))
        for i in range(startidx, len(num)):
            if i != startidx and num[i] == num[i - 1]:  # 保证了子集不会重复
                continue
            subset.append(num[i])
            self.helper(subset, i + 1, num, result)  # 因为每个数只能用一次，因此startidx要从下一位开始 也就是变成i+1
            subset.pop()


import queue


class WordLadder():
    # 给一个包含许多等长的单词的列表dict，如果两个单词只差一个字母则视为neighbor，问从单词a到单词b的最短路径是什么？
    # 既然问最短路径 说明要用BFS
    def ladder_length(self, start, end, dict):
        """
        :param start:起始单词
        :param end: 终止单词
        :param dict: 集合
        :return:
        """
        if dict == None:
            return 0
        if start == end:
            return 1
        dict.add(start)
        dict.add(end)
        hash = set()
        q = queue.Queue()
        q.put(start)
        hash.add(start)

        length = 1
        while not q.empty():
            length += 1
            size = q.qsize()
            for i in range(size):
                word = q.get()
                next_words = self.get_next_words(word, dict)  # 并没有把原来的dict变成图结构，而是用函数现成的找出所有next_words
                for next_word in next_words:
                    if next_word in hash:
                        continue
                    if next_word == end:
                        return length
                    hash.add(next_word)
                    q.put(next_word)
        return 0

    def replace(self, s, index, c):
        new_s = ''
        for i in len(s):
            if i == index:
                new_s += c
            else:
                new_s += s[i]
        return new_s

    def get_next_words(self, word, dict):
        next_words = []
        characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']
        for c in characters:
            for i in range(len(word)):
                if c == word[i]:
                    continue
                next_word = self.replace(word, i, c)
                if next_word in dict:
                    next_words.append(next_word)
        return next_words
class WordLadderII():
    '''与上题类似，但问题是找出所有最短路径，不是返回最短路径的长度，而是返回包含所有最短路径的列表'''

    # 所谓的dfs和bfs结合并不是同时使用 而是先用bfs找出所有节点与终点的距离
    # 再用dfs从起点向距离减小的方向搜索  在子路径添加新节点的过程中，首先要判断要加入的节点是否离当前节点到终点的距离又进了一步
    # 若近了一步 则加入路径中 否则跳过，这样就保证了搜索得到的路径一定是最短路径，不会出现绕远的情况
    # 综上 bfs找出所有节点与终点的距离，这些距离信息在dfs时用来筛选绕远路的节点，从而保证得到的所有路径都是最短路径
    def findLadders(self,start,end,dict):
        dict.add(start)
        dict.add(end)
        distance={}
        # bfs用来查出每个节点与终点end之间的距离 存在distance中
        self.bfs(end,distance,dict)
        results=[]
        self.dfs(start,end,distance,dict,[start],results)
        return results
    def get_next_words(self,word,dict):
        words=[]
        for i in range(len(word)):
            for c in 'abcdefghigklmnopqrstuvwxyz':
                next_word=word[:i]+c+word[i+1:]
                if next_word!=word and next_word in dict:
                    words.append(next_word)
        return words
    def bfs(self,start,distance,dict):
        distance[start]=0
        q=queue.Queue()
        q.put(start)
        while not q.empty():
            word=q.get()
            for next_word in self.get_next_words(word,dict):
                if next_word not in distance:
                    distance[next_word]=distance[word]+1
                    q.put(next_word)

    def dfs(self,curt,target,distance,dict,path,results):
        if curt==target:
            results.append(list(path))
            return
        for word in self.get_next_words(curt,dict):
            if distance[word]!=distance[curt]-1:
                continue
            path.append(word)
            self.dfs(word,target,distance,dict,path,results)
            path.pop()


if __name__ == '__main__':
    # a = Soulution1()
    # c = a.combination_sum([1, 1, 2, 3, 5, 8], 3)
    # print(c)

    # x=Solution2()
    # c=x.palindrome_patitioning('aabccabba')
    # print(c)

    # x = Solution3()
    # c = x.permutations([1, 2, 3])
    # print(c)

    # x=Solution5()
    # res=x.q_queens(4)
    # print(res)
    # a = [1, 2, 3]
    # c = a
    # a = a[:1]
    # print(c)
    a=Solution1()
    candidate=[1,2,2,3,4,4,5]
    target=5
    c=a.combination_sum(candidate,target)
    print(c)
