class Solution_leet1007(object):
    '''
    给定一个由若干 0 和 1 组成的数组 A，我们最多可以将 K 个值从 0 变成 1 。返回仅包含 1 的最长（连续）子数组的长度。
    '''
    def longestOnes(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        left = 0
        right = 0
        count = 0
        res = 0
        ifright_forward = True
        while right < len(A):
            if A[right] == 1:
                right+=1
                res = max(res,right-left)
                ifright_forward = True
            else:
                if ifright_forward:
                    count += 1

                if count<=K:
                    right+=1
                    res = max(res, right-left)
                    ifright_forward = True
                else:
                    if A[left] == 0:
                        count -= 1
                    left += 1
                    ifright_forward = False
        return res
    def longestOnes_v2(self,A,K):
        left = 0
        right = 0
        count = 0
        res = 0
        while right < len(A):
            if A[right] == 1:
                right += 1
                res = max(res, right - left)
            else:
                count += 1

                while count > K:
                    if A[left] == 0:
                        count -= 1
                    left += 1
                right+=1

                res = max(res, right - left)

        return res

class Solution_leet718(object):
    '''给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，
    总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。
    '''
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        left = 0
        right = 0
        num_record = {}
        res = 0
        for i in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',"O",'P','Q','R','S','T','U','V','W','X','Y','Z']:
            num_record[i] = 0
        while right < len(s):
            num_record[s[right]] += 1
            while max(num_record.values())+k<right-left+1:
                a=s[left]
                num_record[a] -= 1
                left+=1
            right+=1
            res = max(res,right-left)
        return res

if __name__ == '__main__':
    a= Solution_leet1007()
    k=a.longestOnes_v2([1,1,1,0,0,0,1,1,1,1,0],2)
    print(k)