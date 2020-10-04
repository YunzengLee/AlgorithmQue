'''关于字符串处理的题'''
'''
这些问题一般没有通俗算法，只是对字符串的特殊处理'''
'''面试58-i 反转字符串的顺序
例子1：
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
示例2：
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
'''
class Solution_mianshi_58_i(object):  # *****
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        length = len(s)
        sublen = 0
        ans = ''
        for i in range(length-1,-1,-1):
            if s[i] !=' ':
                sublen += 1
            elif s[i] == ' 'and sublen!=0:
                ans += s[i+1:i+1+sublen] + ' '# 如果s[i]这个空格在字符串开头，运行这句后ans的尾部就会多一个空格，所以返回结果时要去掉。
                sublen = 0
        if sublen!=0:
            ans += s[0:sublen]
        return ans.strip()
    def reverseWords2(self,s):
        l = s.split(' ')
        for i in l:
            if not i:
                l.remove(i)
        print(l)
        return ' '.join(l[::-1])

if __name__ == '__main__':
    a=Solution_mianshi_58_i()
    res = a.reverseWords2('ojh   ugbugy  uggy kj')
    print(res)