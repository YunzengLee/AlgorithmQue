# ##  动态规划本质上是记忆化搜索，可以通过python的闭包，装饰器，或者默认参数设为列表等不可变对象
# 来存储记忆数据 例如
import functools


def memo(func):
    cache = {}
    @functools.wraps(func)
    def wrap(args):
        # print(cache)
        if args not in cache:
            cache[args] = func(args)
        return cache[args]

    return wrap


@memo
def fib(i):  # 将记忆放在装饰器里
    if i < 2:
        return 1
    return fib(i - 1) + fib(i - 2)


def fib1(n):  # 闭包写法
    l = [0, 1]

    def helper():
        l.append(l[-1] + l[-2])
        return l[-1]

    return helper


def fib1_1(n):  # 闭包写法
    l = [0, 1]

    def helper():
        for i in range(n):
            a = l[-1]
            l.append(l[-1] + l[-2])
        return l[a]

    return helper


def fib2(n, l={1: 1, 2: 1}):  # 默认参数设为不可变对象，虽然正常编程应避免这种情况，
    # 但我们正是使用要这种特性
    if n in l:
        return l[n]
    else:
        l[n] = fib2(n - 1) + fib2(n - 2)
        return l[n]


def fib3():  # 生成器写法
    l = [0, 1]
    while True:
        yield l[-1]
        l.append(l[-1] + l[-2])


class Triangle():
    class DFS_tranverse():
        def __init__(self, A):
            self.best = float('inf')
            self.A = A

        def tranverse(self):
            best = float('inf')
            self.helper(0, 0, 0)
            return best

        def helper(self, x, y, sum):
            if x == n - 1:
                if sum < self.best:
                    self.best = sum
                    return
                return
            self.helper(x + 1, y, sum + self.A[x][y])
            self.helper(x + 1, y + 1, sum + self.A[x][y])
        # 总结 best可以设成全局变量，也可以像这样代入每一层
        #  不对！！！！ best是一个数，是不可变类型，带入函数并不会改变它的值！！
        #
        # 对于一个n层的三角形 里面节点个数为 n^2  (1+2+3+...+n)是n^2的复杂度
        # 对于一个n层的二叉树 里面节点个数为 2^n  因为是翻倍增长，所以最后一层节点数大约是这个问题的规模
        # 上面这个方法的缺陷在于：它的复杂度是O(2^n) 因为它把这个三角形当成了二叉树去遍历 有重复遍历的节点

    class DFS_divide_C():
        def __init__(self, A):
            self.A = A

        def divide_conquer(self, x, y):
            if x == n - 1:  # 如果是最后一层 就直接返回该节点值
                return self.A[x][y]
            left = self.divide_conquer(x + 1, y)
            right = self.divide_conquer(x + 1, y + 1)
            return min(left, right) + self.A[x][y]
        # #很遗憾 这个方法还是O(2^n)  节点是有重复遍历的，还是当成了二叉树

    class DFS_divide_conquer_and_memory():
        # 对上个方法进行优化 用hash存储已经计算的结果，避免重复计算
        # 这样复杂度就变成了O(n^2)

        def answer(self):
            hash = float('inf')
            return self.divide_conquer(0, 0)

        def divide_conquer(self, x, y):
            if x == n - 1:
                return A[x][y]
            if hash[x][y] != float('inf'):
                return hash[x][y]
            hash[x][y] = A[x][y] + min(self.divide_conquer(x + 1, y), self.divide_conquer(x + 1, y + 1))
            return hash[x][y]

    class duo_chong_xun_huan_DP():
        # f[i][j]表示从i,j出发到最后一层的最小路径的长度
        def answer(self):
            # 初始化 最后一层先有值
            for i in range(n):
                f[n - 1][i] = A[n - 1][i]
            for i in range(n - 1, -1, -1):  # i 从n-1到0
                for j in range(i + 1):  # j从0到i
                    f[i][j] = min(
                        f[i + 1][j],
                        f[i + 1][j + 1]
                    ) + A[i][j]
        # 结果就是f[0][0]的值
        # 这是自底向上顺序  下面是自顶向下的顺序


class MinPathSum():
    def min_path_sum(self, grid):
        if grid is None or len(grid) == 0 or len(grid[0]) == 0:
            return 0
        M = len(grid)
        N = len(grid[0])
        sum = [[0 for i in range(M)] for j in range(N)]
        for i in range(1, M):
            sum[i][0] = sum[i - 1][0] + grid[i][0]
        for i in range(1, N):
            sum[0][i] = sum[0][i - 1] + grid[0][i]

        for i in range(1, M):
            for j in range(1, N):
                sum[i][j] = min(sum[i - 1][j], sum[i][j - 1]) + grid[i][j]
        return sum[M - 1][N - 1]


class UniquePath():
    # 求方案总数
    def unique_path_num(self, m, n):
        if m == 0 and n == 0:
            return 1
        sum = [[0 for i in range(m)] for j in range(n)]
        for i in range(m):
            sum[i][0] = 1
        for j in range(n):
            sum[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                sum[i][j] = sum[i - 1][j] + sum[i][j - 1]
        return sum[m - 1][n - 1]


class Climb_Stairs():
    # 小明从第0层走到第n层，每次可以走一层 或走两层 问走到n层有几种走法
    def climb_stair(self, n):
        f = [0 for i in range(n)]
        f[0] = 1
        f[1] = 2
        for i in range(3, n):
            f[i] = f[i - 1] + f[i - 2]
        return f[n - 1]


class LongestIncresingSubsequence():
    # 给一个数据序列，找出最长的递增子序列 返回长度
    def longest_incre_subseq(self, nums):
        f = [1 for i in range(len(nums))]
        max = 0
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    f[i] = max(f[i],f[j]+1) #f[i] if f[i] > f[j] + 1 else f[j] + 1
            if f[i] > max:
                max = f[i]
        return max


class LargestDivisibleSubset():
    # 给一个集合，找出一个子集，子集中任意两个数能整除,返回最大子集的长度
    def solution(self, s):
        s.sort()
        dp = [1 for i in range(s)]
        for i in range(1, len(s)):
            for j in range(i):
                if s[i] % s[j] == 0:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
def test(*nums,**kw):
    print(type(kw))
    print(kw)
    print(type(nums))
    for i in nums:
        print(i)
if __name__ == '__main__':
    test(1,2,3)
    print(type([1,2,3]))