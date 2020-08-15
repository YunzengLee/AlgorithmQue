'''位运算之类的题'''
'''对于0到n之间的数字，求每个数字的二进制有多少个1
比如输入4 输出[0,1,1,2,1]'''


# 思路：数字  011100100
#      减一  011100011
#      与运算011100000
# 任何一个数x与x-1进行与运算后，得到的是去掉了二进制中最后一个1后的x

def getOneNum(x):
    res = [0 for i in range(x + 1)]
    print(len(res))
    print(res)
    for i in range(1, x + 1):
        print(i & (i - 1))
        res[i] = res[i & (i - 1)] + 1
    return res


'''判断一个数是不是2的整数次幂
比如 2 4 8 16
'''


# 解：2的整数次幂在二进制中只有一个1
# 任何一个数x与x-1进行与运算后，得到的是去掉了二进制中最后一个1后的x
# 对于一个2的整数次幂来说，去掉2进制中最后一个1一定就变成0了
def isTwoZheng(x):
    return x & (x - 1) == 0


'''leetcode 面试题56-i  
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。
请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。'''


class Solution_mianshi_56_i(object):
    def singleNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = 0
        for i in nums:
            res ^= i
        # res为a与b的异或
        mask = 1
        while mask & res == 0:
            mask = mask << 1
        # mask是在寻找a b两个数字的不同位

        target1 = 0
        target2 = 0
        for i in nums:
            if i & mask == 0:
                target1 ^= i
            else:
                target2 ^= i
        # 将nums分为两组，一组包含a，另一组包含b，然后就分别在这两组里找唯一数字即可
        return [target1, target2]


if __name__ == '__main__':
    i = 0
    res = 0
    while res < 500:
        i += 1
        res = res + i
    print(res,i)
