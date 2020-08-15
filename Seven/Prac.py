def move_zero(A):
    # 右指针需要指到下一个不为0的值 但左指针只需要从头开始即可 不需要特意指向数组中的第一个0
    left=0
    right=0
    while right<len(A):
        if A[right]!=0:
            tem=A[left]
            A[left]=A[right]
            A[right]=tem
            left+=1
        right+=1
    return A
def duplicate_num(A):
    left=0
    right=0
    s=set()
    while right<len(A):

        right+=1
def two_sum_unique_pair(A,target):
    A.sort()
    print(A)
    left=0
    right=len(A)-1
    res=[]
    while left<right:
        if A[left]+A[right]==target:
            if left != 0 and A[left]==A[left-1]:
                left+=1
                continue
            if right!=len(A)-1 and A[right]==A[right+1]:
                right-=1
                continue
            res.append([A[left],A[right]])
            left+=1
            right-=1
            # print(left,right)
        elif A[left] + A[right] < target:
            left+=1
        else:
            right-=1
    return res
def three_sum(A):
    '''a+b+c=0  b+c=-a'''
    def two_sum(sorted_A,start,end,target,res):
        left=start
        right=end
        # res=[]
        while left<right:
            print(left,right)
            print(-target,sorted_A[left],sorted_A[right])
            if sorted_A[left]+sorted_A[right]==target:
                if left!=start and sorted_A[left]==sorted_A[left-1]:
                    print('#')
                    left+=1
                    continue
                if right!=end and sorted_A[right]==sorted_A[right+1]:
                    print('##')
                    right-=1
                    continue
                print(left, right)
                print(-target, sorted_A[left], sorted_A[right])
                res.append([-target,sorted_A[left],sorted_A[right]])
                # 这里要对left和right改变 忘了这事了
                left+=1
                right-=1
            elif sorted_A[left]+sorted_A[right]<target:
                left+=1
            else:
                right-=1
        return res
    A.sort()
    res=[]
    for i in range(len(A)-2):
        if i!=0 and A[i]==A[i-1]:
            continue
        two_sum(A,i+1,len(A)-1,-A[i],res)
    return res

def triangle_count(A):
    def num(A,start,end,target):
        num=0
        left=start
        right=end
        while left<right:
            if A[left]+A[right]>=target:
                num+=1
                # left+=1
                right-=1
            else:
                left+=1
        return num
    A.sort()
    num_=0
    for i in range(len(A)-1,1,-1):
        num_+=num(A,0,i-1,A[i])
    return num_
        # pass

def two_sum_closest_to_target(A,target):
    res=[]
    if A is None or len(A)<2:
        return res

    A.sort()
    start=0
    end=len(A)-1
    best=float('inf')
    while start<end:
        if abs(A[start]+A[end]-target)<best:
            best=abs(A[start]+A[end]-target)
            res=[A[start],A[end]]
        if A[start]+A[end]>target:
            end-=1
        elif A[start]+A[end]<target:
            start+=1
        else:
            return [A[start],A[end]]
    return res
def three_sum_closest_to_target(A,target):
    def two_sum(A,first_num,start,end,target,best,res):
        while start<end:
            if abs(first_num+A[start]+A[end]-target)<best:
                best=abs(first_num+A[start]+A[end]-target)
                res=[first_num,A[start],A[end]]
            if first_num+A[start]+A[end]==target:
                best=0
                res=[first_num,A[start],A[end]]
            elif first_num+A[start]+A[end]<target:
                start+=1
            else:
                end-=1

    res=None
    if A is None or len(A)<3:
        return res
    A.sort()
    best=float('inf')
    for i in range(len(A)):
        two_sum(A,A[i],i+1,len(A)-1,target,best,res)
    return res
def two_diff_target(A,target):
    res=None
    if A is None or len(A)<2:
        return res
    A.sort()
    left=0
    right=1
    while left<right:
        if A[right]-A[left]==target:
            res=[A[left],A[right]]
            return res
        elif A[right]-A[left]<target:
            right+=1
        else:
            left+=1
        if left==right:
            right+=1


def partition_array(nums,k):
    left=0
    right=len(nums)-1
    while left<right:
        while right>left and nums[left]<k:
            left+=1
        while right>left and nums[right]>=k:
            right-=1
        if left<right:
            tem=nums[left]
            nums[left]=nums[right]
            nums[right]=tem
            left+=1
            right-=1
    if nums[left]<k:
        return left+1
    else:
        return left
def fast_sort(nums):
    def helper(nums,startidx,endidx):
        left=startidx
        right=endidx
        if startidx>=endidx:
            return
        key=nums[left]
        while left<right:
            while left<right and nums[right]>=key:
                right-=1
            nums[left]=nums[right]
            while left<right and nums[left]<=key:
                left+=1
            nums[right]=nums[left]
        nums[left]=key
        helper(nums,startidx,left-1)
        helper(nums,left+1,endidx)


    if nums is None or len(nums)<2:
        return nums
    left=0
    right=len(nums)-1
    helper(nums,left,right)
    return nums
def sansepaixu(nums):
    left=0
    right=len(nums)-1
    i=0
    while i<=right:
        if nums[i]==0:
            nums[i],nums[left]=nums[left],nums[i]
            left+=1
            i+=1
        elif nums[i]==2:
            nums[i],nums[right]=nums[right],nums[i]
            right-=1
        else:
            i+=1
    return  nums

def kth_smallest_num(nums,k):
    def helper(nums, start, end, k):
        left=start
        right=end
        key=nums[left]
        while left<right:
            while left<right and nums[right]>=key:
                right-=1
            nums[left]=nums[right]
            while left<right and nums[left]>=key:
                left+=1
            nums[right]=nums[left]
        nums[left]=key
        if left==k:
            return key
        elif left<k:
            return helper(nums,left+1,end,k)
        else:
            return helper(nums,start,left-1,k)
    if nums is None or len(nums)<k:
        return None
    return helper(nums,0,len(nums)-1,k-1)



if __name__=='__main__':
    # A=[0,1,1,2,2,9,3,3,5]
    # A=[-1,0,1,2,-1,-4]
    # A=[-2,0,1,1,2]
    # print(three_sum(A))
    # print(two_sum_unique_pair(A,2))
    # for i in range(9,1,-1):
    #     print(i)
    # for
    a=[1,0,2,-1,2,3,6,6,4,7]
    # print(partition_array(a,7))
    print(fast_sort(a))