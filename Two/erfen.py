#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""'''二分法'''"""
__author__ = 'LiYunzeng'
num = 100
fengedian = 48
a = list(range(num))
b = a[fengedian:] + a[:fengedian]
print(b)
target = int(input('a number:'))
start = 0
end = len(b) - 1
while start + 1 < end:
    mid = int((start + end) / 2)
    if b[mid] >= b[start]:
        if target >= b[start] and target <= b[mid]:
            end = mid
        else:
            start = mid
    else:
        if target >= b[mid] and target <= b[end]:
            start = mid
        else:
            end = mid
if b[start] == target:
    print(start)
if b[end] == target:
    print(end)


class Solution1():
    # 0 1 矩阵中1是连通的 找出能覆盖所有1的最小矩形
    def find_min_juxing(self, A, x, y):
        if A is None:
            return None
        if len(A) == 0:
            return None

        # 四个边界
        zuo_x_boundry = None
        you_x_boundary = None
        zuo_y_boundary = None
        you_y_boundary = None

        x_max = len(A)
        y_max = len(A[0])
        start = 0
        end = x
        while start + 1 < end:
            mid = (start + end) // 2
            if self.find_1(A, x=mid):  # 如果mid这一行有1
                end = mid
            else:
                start = mid
        if self.find_1(A, x=start):
            zuo_x_boundry = start
        if self.find_1(A, x=end):
            zuo_x_boundry = end

        start = x
        end = x_max
        while start + 1 < end:
            mid = (start + end) // 2
            if self.find_1(A, x=mid):  # 如果mid这一行有1
                end = mid
            else:
                start = mid
        if self.find_1(A, x=start):
            you_x_boundry = start
        if self.find_1(A, x=end):
            you_x_boundry = end

        start = 0
        end = y
        while start + 1 < end:
            mid = (start + end) // 2
            if self.find_1(A, y=mid):  # 如果mid这一行有1
                end = mid
            else:
                start = mid
        if self.find_1(A, y=start):
            zuo_y_boundry = start
        if self.find_1(A, y=end):
            zuo_y_boundary = end

    def find_1(self, A, x=None, y=None):
        is_find = False
        if y == None:
            for j in range(len(A[0])):
                if A[x][j] == 1:
                    is_find = True

        if x == None:
            for i in range(len(A)):
                if A[i][y] == 1:
                    is_find = True
        return is_find


class Solution2():
    # 在先增后减数列中找最大值
    def find_max(self, A):
        # A.sort()
        start = 0
        end = len(A)
        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] < A[mid + 1]:
                start = mid
            if A[mid] > A[mid + 1]:
                end = mid
        if A[start] < A[end]:
            return A[end]
        else:
            return A[start]


class Solution3():
    # 在数列中找出一个极大值，该数列最前一段上升 最后一段下降
    def find_(self, A):
        start = 0
        end = len(A)
        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] > A[mid - 1] and A[mid + 1] > A[mid]:
                start = mid
            elif A[mid] < A[mid - 1] and A[mid + 1] < A[mid]:
                end = mid
            elif A[mid] < A[mid - 1] and A[mid] < A[mid + 1]:
                start = mid
            else:
                return A[mid]
        return A[start] if A[start] > A[end] else A[end]


class Solution4():
    # 在一个切分数列A中找target
    def find_(self, A, target):
        length = len(A)
        if target >= A[0]:
            start = 0
            end = length - 1
            while start + 1 < end:
                mid = (start + end) // 2
                if A[mid] < A[0]:
                    end = mid
                elif A[mid] <= target:
                    start = mid
                else:
                    end = mid
            if A[start] == target:
                return start
            else:
                return end
        if target <= A[length - 1]:
            start = 0
            end = length - 1
            while start + 1 < end:
                mid = (start + end) // 2
                if A[mid] > A[length - 1]:
                    start = mid
                elif A[mid] <= target:
                    start = mid
                else:
                    end = mid
            if A[start] == target:
                return start
            else:
                return end
