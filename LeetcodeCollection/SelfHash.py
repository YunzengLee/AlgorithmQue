'''原地hash'''
'''
leetcode442 一个数组有若干数字，范围是1-n，找出所有重复数字，要求时间复杂度O(n),空间复杂度O(1)
'''
class Solution_442(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''注意数字只能是1-n之间，因此这些数字可以作为下标，
        某一个数字出现一次后，就将该数字所指向的位置的数字加n，来记录该数字出现的次数
        也可以将指向位置的数字取负来表示该数字出现过'''
        n=len(nums)
        for num in nums:
            idx = num%n-1 #该数字所指向位置的下标
            nums[idx]+=n
        res = []
        for idx in range(n):
            if nums[idx]>2*n: # 不能取等号，因为原数字可能是n
                res.append(idx+1)
        return res