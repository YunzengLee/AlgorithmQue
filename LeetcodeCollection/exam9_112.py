#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 输入N和字符S
# @param N int整型 第N+1个元素是前面N个元素之和， N>=2, N<=5
# @param S string字符串 需要解析的字符串，字符串长度不超过1000
# @return int整型一维数组
#
class Solution:
    def split_into_list(self, N, S):
        # write code here
        res = []
        if len(S) == N:
            for i in S:
                res.append(int(i))
                return res
        else:
            pre_N = []
            self.dfs(0, [], N, S, pre_N)
            if not pre_N:
                return []
            for sub in pre_N:
                res = list(map(int,sub))
                length = 0
                for num in sub:
                    length += len(str(num))
                startidx = length
                endidx = length
                while endidx < len(S):
                    if int(S[startidx:endidx + 1]) == sum(res[-N:]):
                        res.append(int(S[startidx:endidx + 1]))
                        startidx = endidx + 1
                        endidx += 1
                    else:
                        endidx += 1
                if startidx == endidx and startidx == len(S):
                    return res

    def dfs(self, startidx, subset, N, S, res):
        if len(subset) == N + 1:
            subset = list(map(int,subset))
            if sum(subset[:-1]) == subset[-1]:
                res.append(list(subset))
            return
        else:
            if startidx < len(S):
                for i in range(startidx + 1, len(S) + 1):
                    subset.append((S[startidx:i]))
                    self.dfs(i, subset, N, S, res)
                    subset.pop()
a=Solution()
res = a.split_into_list(2,'11111')
print(res)