
def merge_sort(nums):
    left=0
    right=len(nums)-1
    return _helper(0,right,nums)

def _helper(startidx, endidx, nums):
    if startidx == endidx:
        return [nums[startidx]]
    mid = (startidx + endidx) // 2
    left = _helper(startidx, mid, nums)
    right = _helper(mid + 1, endidx, nums)
    res = []
    left_p = 0
    right_p = 0
    while left_p < len(left) and right_p < len(right):
        if left[left_p] <= right[right_p]:
            res.append(left[left_p])
            left_p+=1
        else:
            res.append(right[right_p])
            right_p += 1
    while right_p < len(right):
        res.append(right[right_p])
        right_p+=1
    while left_p < len(left):
        res.append(left[left_p])
        left_p += 1
    return res
class MergeSort():
    # 用递归的归并排序
    def m_sort(self, arr):
        """
        对数组arr排序，返回排序后的数组
        :param arr:
        :return:
        """
        if arr is None or arr == [] or len(arr) == 1:
            return arr
        length = len(arr)
        mid = int(length / 2)
        first = arr[:mid]
        last = arr[mid:]
        result = self.sort_algorithm(first, last)
        return result

    def sort_algorithm(self, first, last):
        """
        将两个数组 分别归并排序，最后归并
        :param first:
        :param last:
        :return:
        """

        if len(first) > 1:
            mid_first = int(len(first) / 2)
            first = self.sort_algorithm(first[:mid_first], first[mid_first:])
        if len(last) > 1:
            mid_last = int(len(last) / 2)
            last = self.sort_algorithm(last[:mid_last], last[mid_last:])
        return self.merge(first, last)

    def merge(self, first, last):
        i, j = 0, 0
        res = []
        while i < len(first) and j < len(last):
            if first[i] < last[j]:
                res.append(first[i])
                i += 1
            else:
                res.append(last[j])
                j += 1
        if i == len(first):
            while j < len(last):
                res.append(last[j])
                j += 1
        else:
            while i < len(first):
                res.append(first[i])
                i += 1
        return res


class NoRecurMergeSort():
    # 非递归归并排序
    pass


class FastSort():
    # 快速排序
    def fast_sort(self, arr):
        if arr is None:
            return None
        # if len(arr) == 0 or len(arr) == 1:  # 这两句没必要
        #     return arr
        low = 0
        high = len(arr) - 1
        self.sort_algorithm(arr, low, high)
        return arr

    def sort_algorithm(self, arr, low, high):
        if low < high:
            pivot = self.partition(arr, low, high)
            self.sort_algorithm(arr, low, pivot - 1)
            self.sort_algorithm(arr, pivot + 1, high)

    def partition(self, arr, low, high):
        pivotkey = arr[low]
        while low < high:
            while low < high and arr[high] >= pivotkey:
                high -= 1
            self.swap(arr, high, low)
            while low < high and arr[low] <= pivotkey:
                low += 1
            self.swap(arr, high, low)
        return low

    def swap(self, arr, idx1, idx2):
        tem = arr[idx1]
        arr[idx1] = arr[idx2]
        arr[idx2] = tem

        return 0


def fast_sort(arr):
    low = 0
    high = len(arr) - 1
    q_sort(arr, low, high)
    return arr


def q_sort(arr, low, high):
    if low < high:
        zhongdian = partition(arr, low, high)
        q_sort(arr, low, zhongdian - 1)
        q_sort(arr, zhongdian + 1, high)
    return arr


def partition(arr, low, high):
    povitkey = arr[low]
    while low < high:
        while low < high and arr[high] >= povitkey:
            high -= 1
        swap(arr, low, high)
        # print(arr)
        while low < high and arr[low] <= povitkey:
            low += 1
        swap(arr, low, high)
        # print(arr)
    return low


def swap(arr, idx1, idx2):
    tem = arr[idx1]
    arr[idx1] = arr[idx2]
    arr[idx2] = tem


class NewMergeSort():
    # 上面的写法每次递归都要分出两个数组，数组切分就是个新的空间，太费空间了，能不能用下标值 代替切分数组呢
    def m_sort(self, arr):
        if arr is None:
            return None
        if len(arr) == 0:
            return arr
        start_idx = 0
        end_idx = len(arr) - 1
        result = self.sort_algorithm(arr, start_idx, end_idx)
        return result

    def sort_algorithm(self, arr, start_idx, end_idx):
        if start_idx == end_idx:
            return [arr[start_idx], ]
        mid = (start_idx + end_idx) // 2
        left = self.sort_algorithm(arr, start_idx, mid)
        right = self.sort_algorithm(arr, mid + 1, end_idx)
        res = self.merge(left, right)
        return res

    def merge(self, left, right):
        i, j = 0, 0
        res = []
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        if i == len(left):
            while j < len(right):
                res.append(right[j])
                j += 1
        else:
            while i < len(left):
                res.append(left[i])
                i += 1
        return res

'''非递归快排，递归用栈结构代替'''
# 栈里保存要进行排序的起始终止idx，排序后判断基准位置是否处于起始于终止idx之间，若是，则压入两对新的起始终止idx。

def fastSort_no_recursion(nums):
    stack=[]
    if nums is None or len(nums)<2:
        return nums
    left=0
    right=len(nums)-1

    stack.append((left,right))
    while len(stack)!=0:
        boundary = stack.pop()
        par = partition(nums,boundary[0],boundary[1])
        if par-1>boundary[0]:
            stack.append((boundary[0],par-1))
        if par+1<boundary[1]:
            stack.append((par+1,boundary[1]))
    return nums






if __name__ == '__main__':
    arr = [5, 6, 78, 2, 3, 1, 5, 7]
    print(fastSort_no_recursion(arr))
    # print(len(arr) // 2)
    # print(fast_sort(arr))
    # a='a'
    # print(ord(a))
    # a = NewMergeSort()
    # c = a.m_sort(arr)
    # print(c)
