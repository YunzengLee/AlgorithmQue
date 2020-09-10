'''
单调栈
'''

'''
leetcode 739 每日温度
请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。
例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
注意：栈内存的是下标，不是温度值。要灵活运用!!!!!!!!!
'''
class Solution_leet739(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        res = [0 for i in range(len(T))]
        stack = []
        cur_idx = 0
        while cur_idx<len(T):
            if not stack:
                stack.append(cur_idx)
            else:
                while stack and T[stack[-1]]<T[cur_idx]:
                    pop_idx = stack.pop()
                    res[pop_idx] = cur_idx - pop_idx
                stack.append(cur_idx)
            cur_idx+=1
        return res



''''''
'''面试题59-ii 队列的最大值'''
# 实现一个队列类 可以以O（1）的复杂度取最大值
# 本质上还是使用辅助队列的思想，与 取栈的最大值 的做法类似。
# 本题用到是双端队列 需要根据情况从左边或者右边pop出值
import queue


class MaxQueue:

    def __init__(self):
        self.deque = queue.deque()
        self.queue = queue.Queue()

    def max_value(self) -> int:
        return self.deque[0] if self.deque else -1

    def push_back(self, value: int) -> None:
        while self.deque and self.deque[-1] < value:
            self.deque.pop()  # 用while循环把前面的小于该数的都去掉了
        self.deque.append(value)
        self.queue.put(value)
        print(self.deque)
        # print(self.queue)
        print(' ')

    def pop_front(self) -> int:
        if not self.deque:
            return -1
        ans = self.queue.get()
        if ans == self.deque[0]:
            self.deque.popleft()
        print(self.deque)
        print(self.queue)
        return ans
