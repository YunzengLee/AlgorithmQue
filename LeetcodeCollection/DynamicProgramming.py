'''动态规划问题'''

'''接龙型'''
'''面试题42  主站53 最大连续子数组的最大值'''


# 给一个数组，求这个数组中所有连续子数组的和，返回其中最大值
class Solution53(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        max_val = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
            max_val = max(max_val, dp[i])
        return max_val


'''leetcode198 打家劫舍'''


# 三种解法 三种思路

class Solution198:
    def rob(self, nums):
        if nums is None or nums == []:
            return 0
        if len(nums) <= 2:
            return max(nums)
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = nums[1]
        maxval = max(dp[0], dp[1])
        for i in range(2, len(nums)):
            for j in range(max(0, i - 3), i - 1):
                dp[i] = max(dp[i], nums[i] + dp[j])
            maxval = max(maxval, dp[i])

        return maxval

    def rob2(self, nums):
        # 解法2：将抢或不抢也算入状态中。dp[i][x]代表走到i位置时，抢该位置或不抢该位置，得到的最大value
        n = len(nums)
        if n == 0:
            return 0
        dp = [[0, 0] for i in range(len(nums))]
        dp[0][0] = 0
        dp[0][1] = nums[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
            dp[i][1] = nums[i] + dp[i - 1][0]
        return max(dp[n - 1][0], dp[n - 1][1])

    def rob3(self, nums):
        # 解法3 dp[i]表示走到i时，能抢到的最大value
        dp = [0 for i in nums]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i], dp[i - 1])  # 第i个没有抢
            dp[i] = max(dp[i], dp[max(0, i - 2)] + nums[i])  # 第i个抢了
        return dp[len(nums) - 1]


'''leetcode32 最长有效括号  给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。'''


def tes_t_1(s):
    if not s or len(s) <= 1:
        return 0
    dp = [0 for i in range(len(s))]
    maxval = float('-inf')
    for i in range(len(s) - 1, -1, -1):

        if s[i] == '(':
            continue
        else:
            if i + 1 < len(s) and i + 1 + dp[i + 1] < len(s):
                next_idx = i + 1 + dp[i + 1]
                if s[next_idx] == ')':
                    dp[i] = dp[i + 1] + 2
                    if next_idx + 1 < len(s):
                        dp[i] += dp[next_idx + 1]
                maxval = max(dp[i], maxval)
    return maxval


# 类似于接龙型，只不过是从后向前走，后缀型的接龙。
class Solution32:
    def longestValidParentheses(self, s: str) -> int:
        if len(s) < 2:
            return 0
        dp = [0 for i in range(len(s))]
        # dp[i]代表从第i处往后的最长有效括号长度
        maxval = 0
        n = len(s)
        for i in range(n - 2, -1, -1):
            if s[i] == '(':
                j = i + dp[i + 1] + 1
                if j < n and s[j] == ')':
                    dp[i] = dp[i + 1] + 2
                    if j + 1 < len(s):
                        dp[i] += dp[j + 1]
                maxval = max(maxval, dp[i])
        return maxval


class Solution_holiday:
    '''n天假期，可以工作锻炼或休息，不能连续两天工作，不能连续两天休息，问最少休息几天，给天数n，给公司开门和健身房营业的情况（只能在公司营业时上班，健身房营业时锻炼）'''

    '''dp[i]代表第i天最少休息几天，但这个不够，还需一个维度，表示第i天干了什么事情。
    dp [i] [j] 定义第i天去干第j件事的最少休息天数
    j可以取值0：休息   1：工作   2：锻炼
    最后的答案是什么：是dp[n-1] [0] dp[n-1] [1] dp[n-1] [2]三个数的最小值
    
    这个与打家劫舍解法2类似，dp有两个维度，一个是问题规模n，另一个是固定的。
    '''

    def longestHoliday(self, n, company_list, practice_list):
        dp = [[float('inf'), float('inf'), float('inf')] for i in range(n)]
        dp[0][0] = 1
        dp[0][1] = 0
        dp[0][2] = 0
        for i in range(1, n):
            dp[i][0] = min(dp[i - 1][1], dp[i - 1][2]) + 1
            if company_list[i]:
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][2])
            else:
                dp[i][1] = float('inf')
            if practice_list[i]:
                dp[i][2] = min(dp[i - 1])
            else:
                dp[i][2] = float('inf')
        return min(dp[-1])


'''坐标型'''

'''
动态规划专栏整理：
'''
'''斐波那契数列，青蛙跳台阶问题'''


class Solution_JianZhiOffer10():
    def solution(self, n):
        a, b = 0, 1  # 若是青蛙跳台阶问题则 a=1 b=1
        for _ in range(n):
            a, b = b, a + b  # 属于动态规划算法，只是没有存储更早的结果
            # 这两句也需要仔细体会，是python专用用法，这两个赋值没有先后顺序
        return a


'''leetcode174地下城游戏'''


class Solution174(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        ############# 坐标型dp，难点在于状态的定义，
        # 需要自底向上，状态定义有点难想
        # 一开始想的是自起点到终点，每一位置的状态代表从起点到此处至少需要的血量
        # 实际上自底向上更简单，每一位置定义为从此处到终点至少需要的血量
        m = len(dungeon)
        n = len(dungeon[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        dp[m - 1][n - 1] = max(1, 1 - dungeon[m - 1][n - 1])
        for i in range(m - 2, -1, -1):
            dp[i][n - 1] = max(1, - dungeon[i][n - 1] + dp[i + 1][n - 1])
        for i in range(n - 2, -1, -1):
            dp[m - 1][i] = max(1, - dungeon[m - 1][i] + dp[m - 1][i + 1])
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                dp[i][j] = max(1, - dungeon[i][j] + min(dp[i + 1][j], dp[i][j + 1]))
        return dp[0][0]


'''匹配型，  两个字符串或列表 dp是二维的 但不代表坐标'''
'''面试题60 n个骰子的点数之和'''


class Solution_mianshi_60:
    def twoSum(self, n):
        f = [[0 for i in range(6 * n + 1)] for j in range(n + 1)]

        for i in range(1, 7):
            f[1][i] = 1

        for i in range(2, n + 1):
            for j in range(i, 6 * n + 1):
                for k in range(1, 7):
                    if j > k:
                        f[i][j] += f[i - 1][j - k]

        res = [0 for I in range(6 * n - n + 1)]
        for i in range(0, 6 * n - n + 1):
            res[i] = f[n][n + i] / ((6 ** n))
        return res

class Solution_leetcode718:
    '''求两个数组的公共子数组的最大长度'''
    def findLength(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        '''匹配性动归'''
        lena = len(A)
        lenb = len(B)
        dp = [[0 for i in range(len(B))] for j in range(len(A))]
        res = 0
        for j in range(lenb):
            dp[lena-1][j] = 1 if A[lena-1] == B[j] else 0
            res = max(res,dp[lena-1][j])
        for j in range(lena):
            dp[j][lenb-1] = 1 if A[j] == B[lenb-1] else 0
            res = max(res,dp[j][lenb-1])
        for i in range(lena-2,-1,-1):
            for j in range(lenb-2,-1,-1):
                if A[i] == B[j]:
                    dp[i][j] = dp[i+1][j+1] + 1
                    res = max(res,dp[i][j])
        return res

# leetcode 10  正则匹配 测试中
class Solution_xxx:
    def ifMatch(self, s, p):
        dp = [[False for _ in range(len(p) + 1)] for _ in range(len(s) + 1)]
        dp[0][0] = True
        # for i in range(1, len(s)+1):
        #     dp[i][0] = False
        for i in range(1, len(p) + 1):
            if i % 2 == 0:
                if p[i - 1] == '*':
                    dp[0][i] = dp[0][i - 2]
        for i in range(1, len(s) + 1):
            for j in range(1, len(p) + 1):
                if p[j - 1] != '*':
                    if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                else:
                    if p[j - 2] == '.':  # *前面是.
                        for k in range(i, -1, -1):
                            if dp[k][j - 2]:
                                dp[i][j] = True
                                break
                    else:  # *前面是字母
                        dp[i][j] = dp[i][j - 2]
                        k = i
                        while k > 0 and s[k - 1] == p[j - 2]:
                            if dp[k - 1][j - 2]:
                                dp[i][j] = True
                                break
                            k -= 1
        return dp[-1][-1]


'''背包型 '''


class Solution_stoneMerge:
    """
    n个石头，质量不等，可以让他们相互碰撞抵消（次数不限），直到剩一个（或0个）石头，问剩下的石头最小质量可以是多少？
    输入：n个石头的质量
    类似：背包型动归（特点：无序的，与两数之和（差）有关）
    思路：把石子分两堆，这两堆各自的质量之和要尽量接近
    用j循环从1到石头总质量/2大小的背包
    dp[j] 表示将容量为j的01背包装满是否可行，如果可行，|sum-2*j|就是目前的碰撞结果
    每装满一个大小为x的背包，需要维护|sum-2*x|的最小值
    """

    def stoneMerge(self, stone):
        stone_sum = sum(stone)
        dp = [False for j in range(stone_sum // 2 + 1)]
        dp[0] = True
        print(dp)
        n = len(stone)
        # stone.sort()  # 不必排序，因为先使用和后使用哪个石头是没有区别的，因为结果只跟他们的和有关
        for i in range(n):
            for j in range(stone_sum // 2, stone[i] - 1, -1):  # 必须从高到低遍历，否则如下行代码
                dp[j] = dp[j - stone[i]] or dp[j]
                # for j in range(stone[i], len(dp)):  # 这样写的话，在一个i的for循环中，会出现一个stone[i]被使用多次的情况
                #     print(j)
                #     dp[j] = dp[j - stone[i]] or dp[j]
                print(dp)
            print('#')

        print(dp)
        res = float('inf')
        for j in range(stone_sum // 2, -1, -1):
            if dp[j]:
                res = min(res, abs(j - (stone_sum - j)))
        return res


'''区间型'''


class Solution_coinMerge:
    """
    N堆金币排成一排（数组），第i堆有c【i】个金币，每次将相邻的两堆金币合并，合并成本为两堆金币之和，经过N-1次合并后合并为一堆，求最小成本
    """

    def mergeCoin(self, coins):
        """
        令dp[i][j]表示将第i到j堆合并的所需的最小成本
        状态方程：dp[i][j] = min(dp[i][j],dp[i][k]+dp[k][j]+sum(i,j))  大区间的结果由小区间递推得到，因此最外一层循环是区间长度，顺序是从小到大
        :param coins:
        :return:
        """
        n = len(coins)
        dp = [[0 for j in range(n + 1)] for i in range(n + 1)]
        sum_coin = [0 for i in range(n + 1)]
        for i in range(1, n + 1):
            sum_coin[i] = sum_coin[i - 1] + coins[i - 1]
        for length in range(2, n + 1):  # 区间长度为2到n
            for i in range(1, n - length + 2):
                j = i + length - 1
                dp[i][j] = float('inf')
                least_sum = sum_coin[j] - sum_coin[i - 1]
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + least_sum)
        print(dp[1][n])


class Solution_twoColorTower:
    """
    双色塔
    给红色绿色石头若干，建塔，第一层1个石头，第i层i个石头，每一层石头颜色相同，塔的层数尽可能多，问最多有几种建塔方案
    """

    def twoColerTower(self, red, green):
        if not red or not green:
            return 1
        red, green = min(red, green), max(red, green)
        dp = [0 for i in range(green + 1)]
        totalNeed = 1
        left, right = 0, 0
        dp[0] = 1
        dp[1] = 1
        for i in range(2, int((2 * (red + green)) ** 0.5) + 1):
            totalNeed += i
            maxNeedA = min(totalNeed, red)
            minNeedA = max(totalNeed - green, 0)
            if minNeedA > maxNeedA:
                break
            left = minNeedA
            right = maxNeedA
            for j in range(i, right - 1, -1):
                dp[j] = dp[j] + dp[j - i]
        sum_ = 0
        for i in range(left, right + 1):
            sum_ += dp[i]
        return sum_


if __name__ == '__main__':
    # import re
    #
    # s = '123sdef#21(de'
    # nums = re.findall(r'\d+', s)
    # character = re.findall(r'[a-zA-Z]+', s)
    # print(nums)
    # print(character)
    #
    # a = Solution_stoneMerge()
    # a.stoneMerge([2, 5, 4, 7])
    a = Solution_coinMerge()
    a.mergeCoin([2, 4, 5, 7])
