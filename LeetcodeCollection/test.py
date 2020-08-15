import functools



def gcd(x, y):
    # 最大公约数
    if x == 0:
        return y
    else:
        return gcd(y % x, x)

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
def test(x):
    x=[1,2,3]

if __name__=='__main__':
    import re
    # s='02'
    # res = re.match(r'^[-|+]?\d+[\.[\d]+]?[[e|E][-|+]?\d+]?$', s)
    # if res:
    #     print('yes')
    # else:
    #     pass
    def dfs(numstr, num, res):
        print(numstr)
        if int(numstr) % num == 0:
            res[0] += 1
        if len(numstr) == 1:
            return
        for i in range(1, len(numstr)):
            if int(numstr[:i]) % num == 0:
                dfs(numstr[i:], num, res)


    T = int(input())
    res = []
    for i in range(T):
        n_and_m = input().split()
        n = n_and_m[0]
        m = n_and_m[1]
        m=int(m)
        numstr = input()
        ans = [0]
        dfs(numstr, m, ans)
        res.append(ans[0])
    for i in res:
        print(i)