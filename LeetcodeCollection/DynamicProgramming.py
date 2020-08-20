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
            dp[i] = max(nums[i], dp[i-1]+nums[i])
            max_val = max(max_val, dp[i])
        return max_val
'''leetcode198 打家劫舍'''
#三种解法 三种思路

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
def test1(s):
    if not s or len(s)<=1:
        return 0
    dp = [0 for i in range(len(s))]
    maxval = float('-inf')
    for i in range(len(s)-1, -1,-1):

        if s[i] == '(':
            continue
        else:
            if i+1 < len(s) and i+1 + dp[i+1]<len(s):
                next_idx = i+1+dp[i+1]
                if s[next_idx] == ')':
                    dp[i] = dp[i+1]+2
                    if next_idx+1<len(s):
                        dp[i] += dp[next_idx+1]
                maxval = max(dp[i],maxval)
    return maxval

# 类似于接龙型，只不过是从后向前走，后缀型的接龙。
class Solution32:
    def longestValidParentheses(self, s: str) -> int:
        if len(s)<2:
            return 0
        dp = [0 for i in range(len(s))]
        # dp[i]代表从第i处往后的最长有效括号长度
        maxval = 0
        n=len(s)
        for i in range(n-2, -1,-1):
            if s[i] == '(':
                j = i+dp[i+1]+1
                if j<n and s[j] == ')':
                    dp[i] = dp[i+1] + 2
                    if j+1<len(s):
                        dp[i] += dp[j+1]
                maxval = max(maxval,dp[i])
        return maxval


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
    def twoSum(self,n):
        f=[[0 for i in range(6*n+1)] for j in range(n+1)]

        for i in range(1,7):
            f[1][i] = 1

        for i in range(2,n+1):
            for j in range(i,6*n+1):
                for k in range(1,7):
                    if j>k:
                        f[i][j] += f[i-1][j-k]

        res = [0 for I in range(6*n-n+1)]
        for i in range(0,6*n-n+1):
            res[i] = f[n][n+i]/((6**n))
        return res

# leetcode 10  正则匹配 测试中
class Solution_xxx:
    def ifMatch(self, s, p):
        dp = [[False for _ in range(len(p)+1)] for _ in range(len(s)+1)]
        dp[0][0] = True
        # for i in range(1, len(s)+1):
        #     dp[i][0] = False
        for i in range(1,len(p)+1):
            if i % 2 == 0:
                if p[i-1] == '*':
                    dp[0][i] = dp[0][i-2]
        for i in range(1, len(s)+1):
            for j in range(1,len(p)+1):
                if p[j-1] != '*':
                    if p[j-1] == '.' or p[j-1] == s[i-1]:
                        dp[i][j] = dp[i-1][j-1]
                else:
                    if p[j-2] == '.': # *前面是.
                        for k in range(i,-1,-1):
                            if dp[k][j-2]:
                                dp[i][j] = True
                                break
                    else: # *前面是字母
                        dp[i][j] = dp[i][j-2]
                        k = i
                        while k>0 and s[k-1] == p[j-2]:
                            if dp[k-1][j-2]:
                                dp[i][j] = True
                                break
                            k -= 1
        return dp[-1][-1]


if __name__ == '__main__':
    import re
    s = '123sdef#21(de'
    nums = re.findall(r'\d+', s)
    character = re.findall(r'[a-zA-Z]+',s)
    print(nums)
    print(character)