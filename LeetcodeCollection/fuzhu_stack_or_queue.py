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
