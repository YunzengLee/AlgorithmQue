# 快速排序再写一遍
class Solution1():
    def fast_sort(self,nums):
        return self.sort(nums,0,len(nums)-1)
    def sort(self,nums,start,end):
        left=start
        right=end
        if start>=end:
            return
        key=nums[start]

        while start < end:
            while start<end and nums[end] >= key:
                end-=1
            nums[start]=nums[end]
            while start<end and nums[start] <= key:
                start+=1
            nums[end]=nums[start]
        nums[start]=key
        self.sort(nums,left,start-1)
        self.sort(nums,start+1,right)
        return nums
class Solution2():
    # 正经快排
    def fast_sort(self,nums):
        # print(len(nums)-1)
        self.helper(nums,0,len(nums)-1)
        return nums
    def helper(self,nums,start,end):
        left=start
        right=end
        if start>=end:
            return
        key=nums[start]

        while start<end:
            while start<end and nums[end]>=key:
                end-=1
            self.swap(nums,start,end)
            while start<end and nums[start]<=key:
                start+=1
            self.swap(nums,start,end)
        # print(nums)
        self.helper(nums,left,start-1)
        self.helper(nums,start+1,right)
        # 此时key处于start下标处


    def swap(self,nums,ind1,ind2):
        tem=nums[ind1]
        nums[ind1]=nums[ind2]
        nums[ind2]=tem
if __name__=='__main__':
    s=Solution1()
    arr = [5, 6, 78, 2, 3, 1, 5, 7]
    sort_arr=s.fast_sort(arr)
    print(sort_arr)