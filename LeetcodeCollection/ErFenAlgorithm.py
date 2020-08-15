'''二分法'''
'''
leetcode 面试题53-ii  0-n-1中缺失的数字
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
'''
# 没想到用二分法 ， 另外就是两个特殊情况没有考虑到
class Solution_mianshi_53_ii(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: # 只有一个数的情况
            return 1 if nums[0] == 0 else 0
        # 最右边的数字缺失的情况 ，如果是这种情况，right永远不会找到这个数字
        # 所以要单独处理
        if nums[-1] != len(nums):
            return len(nums)
        left=0
        right=len(nums)-1
        while left+1<right:
            mid = (left+right)//2
            if nums[mid]>mid:
                right = mid
            else:
                left = mid
        if nums[left]==left:
            return right
        else:
            return left