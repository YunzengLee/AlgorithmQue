class Solution1():
    def maxSubarray(self, A):
        # 给一个数组A，求具有最大和的子数组的和（没看透）
        # 我去 这是O(n)的复杂度啊  sum就是前缀和 走一步算一步 没有用列表存储 省了空间
        # 若是返回数组的话 只需要多写两个变量记录下标即可
        # 这题用到了前缀和数组的思想 但节省了前缀和数组的空间
        # 换做我 肯定先用 n的时间复杂度和n的空间复杂度计算前缀和数组 再用n^2复杂度找最大和子数组
        if A is None or len(A) == 0:
            return 0
        max_val = float('-inf')
        sum = 0
        minSum = 0
        for i in range(len(A)):
            sum += A[i]
            max_val = max(max_val, sum - minSum)
            minSum = min(minSum, sum)
        return max_val


class Solution2():
    def findZeroSubarray(self, A):
        # 找出A的子数组中和为0的一个子数组，返回起始位置和终止位置，可能有多个子数组和为0，只找到一个即可
        # 解法：找到前缀和数组PrefixSum，找里面相等的两个元素。
        prefixSum = [0]
        sum = 0
        for i in range(len(A)):
            sum = A[i] + sum
            prefixSum.append(sum)
        # 接下来找prefixSum里相同的两个数的下标

    def findCloseZeroSubarray(self):
        # 找出尽量接近0的子数组
        # 方法： 写出prefixSum数组，找里面最接近的两个数： 做个排序，就可以找出最接近的
        # 另一种方法： 使用TreeMap这种数据结构  还不懂
        pass


## 综上 遇见子数组 和 这样的关键词，基本用prefixSum

#  同向双指针
class Solution3:
    def move_zeros(self, A):
        # 将A中0移到最后
        # left指针指向列表A中非0数的下一个（也就是第一个0） # right指针不断向右遍历
        right = 0
        left = 0
        for right in range(len(A)):
            if A[right] != 0:
                tem = A[right]
                A[right] = A[left]
                A[left] = tem
                left += 1
        return A

    def duplicate_numbers_in_array(self, A):
        # 返回数组A中unique数的个数 把这些unique数放在数组A的前面，重复数放到后面
        # 方法与上题类似  # 没看透
        # left指向已经排好的unique数的最后一个，所以交换前left先加1 right为遍历指针

        A.sort()
        left = 0
        for right in range(len(A)):
            if A[right] != A[left]:
                left += 1
                tem = A[right]
                A[right] = A[left]
                A[left] = tem

        return left + 1


class Solution4():
    # 相向双指针
    def valid_palindrome(self, string):
        # 判断是否为回文串
        left = 0
        right = len(string) - 1
        while left < right:
            if string[left] == string[right]:
                left += 1
                right -= 1
            else:
                return False
        return True


'''Two Sum 问题，一个数组里找等于target的两个数，遍历一个数，再看target减这个数是否存在于数组中'''
'''若是已排序数组就用相向双指针向中间靠拢寻找'''
'''有些问题 如unique pair就需要先排序、再用双指针。unique pair比已排序数组找两数和多了一步：需要用if句跳过不符合的值'''
class TwoSum():
    # twosum: data structure design   设计一个数据结构，实现以下功能
    def __init__(self):
        self.l = []
        self.d = {}

    def add(self, number):
        # 添加一个数进入该结构
        if number in self.d.items():
            self.d[number] += 1
        else:
            self.d[number] = 1
            self.l.append(number)

    def find(self, value):
        # 查看是否已经有一对数 之和等于value
        for i in range(len(self.l)):
            num1 = self.l[i]
            num2 = value - num1
            if num1 == num2 and self.d[num1] > 1:
                return True
            if num1 != num2 and num2 in self.d:
                return True
        return False


# Two Sum 的 follow up
class Solution5():
    def TwoSum_UniquePairs(self, nums, target):
        # 给一个数组，看里面有多少unique pair之和等于target，返回pair的数量
        # unique pair指的是独一无二的一对。若数组中有重复数，也只能用一次
        nums.sort()
        left = 0
        right = len(nums) - 1
        count = 0
        while left < right:
            if nums[left] + nums[right] == target:
                count += 1
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
            elif nums[left] + nums[right] < target:
                left += 1
            else:
                right -= 1
        return count


class ThreeSum():
    def three_sum(self, nums):
        res = []
        if nums is None or len(nums) < 3:
            return res
        nums.sort()
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = len(nums) - 1
            target = -nums[i]
            self.two_sum(nums, left, right, target, res)
        return res

    def two_sum(self, nums, left, right, target, res):
        while left < right:
            if nums[left] + nums[right] == target:
                res.append((-target, nums[left], nums[right]))
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif nums[left] + nums[right] < target:
                left += 1
            else:
                right += 1


class TriangleCount():
    # 给一个数组，给出所有三元组，里面的三个数能组成三角形.返回三元组的个数
    # 组成三角形的条件 ： a<=b<=c and  a+b>c
    # 注意：若数组为[1,1,1,1]，则可以组成4个答案，即四个数中取其中三个。
    def triangle_count(self, nums):
        nums.sort()
        ans = 0
        for i in range(1, len(nums)):
            left = 0
            right = i - 1
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    ans += right - left
                    right-=1
                else:
                    left += 1
        return ans


class TwoSum_ClosesttoTarget():
    # 相向指针
    def two_sum_closest(self, nums, target):
        nums.sort()
        i = 0
        j = len(nums) - 1
        best = float('inf')
        # best_=(,)
        while i < j:
            diff = abs(nums[i] + nums[j] - target)
            best = min(best, diff)

            if nums[i] + nums[j] < target:
                i += 1
            else:
                j -= 1
        return best


class ThreeSumClosesttoTarget():
    pass


class TwoSum_DiffEqualsToTarget():
    # 两数之差等于target的元组  若返回下标，则用dict存储一下原数组
    #  ##两数之差就不能用相向双指针了 要用同向双指针
    def difference_equal_target(self, nums, target):
        nums.sort()
        left = 0
        right = 1
        res = []
        while left < right:
            if nums[right] - nums[left] < target:
                right += 1
            elif nums[right] - nums[left] > target:
                left += 1
            else:
                res.append((nums[left], nums[right]))
                left += 1
            if left == right and right < len(nums) - 1:
                right += 1
        return res


# Partition Array

# 分两部分
class PartitionArray():
    # 把小于k的移到左边 大于等于k的移动到右边，返回排序后第一个大于等于k的值的下标
    def swap(self, nums, idx1, idx2):
        pass

    def partition_array(self, nums, k):
        left = 0
        right = len(nums) - 1
        while left < right:
            while left < right and nums[left] < k:  # 这个while循环避免出现left>right的情况 最多相等 也为了防止left超出数组长度
                left += 1
            # left指向左边第一个不符合条件的
            while left < right and nums[right] >= k:
                right -= 1
            if left < right:  # 前面两个while使两个指针最多重合在一起，这个if条件使重合时不必进行交换 因为重合时交换会导致left和right各自错过
                self.swap(nums, left, right)
                left += 1
                right -= 1
        # left 或left+1就是答案
        if nums[left] < k:
            return left + 1
        return left


# Quick select算法  快速排序两边都要递归 但quick select根据目标在哪只去一边
class KthSmallestNuminUnsortedArr():
    # 找出一个未排序数组中第k小的值
    def kthSmallest(self, k, nums):
        return self.quickselect(nums, 0, len(nums) - 1, k - 1)

    def quickselect(self, A, start, end, k):
        if start == end:
            return A[start]
        left = start
        right = end
        pivot = A[(start + end) // 2]   # 数组中间位置的值做枢轴 其实这个枢轴选哪个位置都无所谓 因为不知道具体值
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[left] > pivot:
                right -= 1
            if left <= right:
                tem = A[left]
                A[left] = A[right]
                A[right] = tem
                left += 1  # 快速排序中，交换之后
                right -= 1
        # 以上与快速排序类似(与快速排序的区别也是有意思的) 以下是算法关键
        # 经过上面的内容 最后的left与right 要么是left=right+1 要么是left=right+2
        # 即left与right已经错开了 二者要么隔一个数 要么相邻  所以下面关于k的位置的判断只有三种情况
        if right >= k and start <= right:
            return self.quickselect(A, start, right, k)
        elif left <= k and left <= end:
            return self.quickselect(A, left, end, k)
        else:
            return A[k]


# 分三部分
class SortColors():
    # nums=[1,2,2,2,1,1,0,0,0] return[0,0,0,1,1,1,2,2,2]
    # 解法：先把0和（1，2）分开，再从0的右边分开1和2。  也就是做两次partition。
    # 标准做法
    def swap(self, a, b, c):
        pass

    def sort_colors(self, a):
        if a is None or len(a) <= 1:
            return
        pl = 0
        pr = len(a) - 1
        i = 0
        while i <= pr:
            if a[i] == 0:
                self.swap(a, i, pl)
                i += 1
                pl += 1
            elif a[i] == 2:
                self.swap(a, i, pr)
                # z这里i不能+1  因为换过来的数可能是0 要再进行一次处理
                pr -= 1
            else:
                i+=1

class SortColorsII():
    # 又叫彩虹排序
    # colors数组（长为n）中共有k个数（k<=n）,对它排序
    # 普通排序算法为O(n * log n) 但这个题目有k这个已知量，以下解法复杂度为O(n * log k)
    def sort_colors2(self, colors, k):
        if colors is None or len(colors) == 0:
            return
        self.rainbowsort(colors, 0, len(colors) - 1, 1, k)

    def rainbowsort(self, colors, left, right, color_from, color_to):
        if color_from == color_to:
            return
        if left >= right:
            return
        color_mid = (color_from + color_to) // 2
        l = left
        r = right
        while l <= r:
            while l <= r and colors[l] <= color_mid:
                l += 1
            while l <= r and colors[r] > color_mid:
                r -= 1
            if l <= r:
                tem = colors[l]
                colors[l] = colors[r]
                colors[r] = tem
                l += 1
                r -= 1
        self.rainbowsort(colors, left, r, color_from, color_mid)
        self.rainbowsort(colors, l, right, color_mid, color_to)


if __name__ == '__main__':
 #    x = Solution3()
 #    c = x.duplicate_numbers_in_array([1, 2, 0, 3, 3, 6, 0, 7, 8])
 #    print(c)
    a=[1,1,2]
    c=a.pop()
    print(c)
    print(a)