class LRUcache_v1():
    # 使用双向指针
    class Node():
        def __init__(self, key, val):
            self.prev = None  # 定义有双指针的节点
            self.next = None
            self.key = key
            self.val = val

    def __init__(self):
        self.head = self.Node(-1, -1)
        self.tail = self.Node(-1, -1)  # 定义头结点和尾节点 其顺序为head->other nodes->tail
        self.hs = {}  # java中这里是hashmap 相当于python中的dict
        self.capacity = None

    def LRUcache(self, capacity):
        self.capacity = capacity
        self.tail.prev = self.head
        self.head.next = self.tail

    def get(self, key):
        if key not in self.hs:
            return -1
        # remove current
        current = self.hs[key]  # 给一个key 找到与key对应的节点current
        # 并将该节点移到队尾
        current.prev.next = current.next  # ######自己写忘了这两句
        current.next.prev = current.prev
        # move current to tail
        self.move_to_tail(current)
        return self.hs[key].val  # 返回key对应的节点的val

    def set(self, key, val):
        # update the position of key in the linked list
        if self.get(key) != -1:
            self.hs[key].val = val
            return
        # 以上是重新设置key对应的节点的val
        # 如果没有该节点就执行以下代码
        if len(self.hs) == self.capacity:
            # 该class内部维护着一个链表当内存队列 以及一个dict用来找对应的node
            # 如果内存已满，则在链表中丢掉最前面的节点 并丢掉dict中对应的存储
            self.hs.pop(self.head.next.key)  # 丢掉dict中的节点 head节点的下一个就是最远使用的节点
            self.head.next = self.head.next.next  # 丢掉链表中的节点
            self.head.next.prev = self.head
        # 创建新节点 存入dict 并放在链表最后
        insert = self.Node(key, val)
        self.hs[key] = insert
        self.move_to_tail(insert)

    def move_to_tail(self, current):
        current.prev = self.tail.prev
        self.tail.prev = current
        current.prev.next = current
        current.next = self.tail
class KeyValue():
    def __init__(self,key,val):
        self.val=val
        self.next=None
        self.key=key
class LRUcache_v2():
    # 使用单指针  # 还没看透 存储位置向后偏移了一位，这样设计可能是为了方便更改
    def __init__(self,capacity):
        self.head=KeyValue(-1,-1)
        self.tail=self.head
        self.hash={}
        self.size=0  # 当前队列长度
        self.capacity=capacity # 最大容量


    def move_to_tail(self,prev):
        if prev.next==self.tail:  # 如果已经是最后一个了 就不执行任何操作  #####自己写忘了这句
            return
        node=prev.next
        prev.next=node.next
        if node.next is not None:
            self.hash[node.next.key]=prev
        self.tail.next=node
        node.next=None
        self.hash[node.key]=self.tail
        self.tail=node
    def get(self,key):
        if key not in self.hash:
            return -1
        self.move_to_tail(self.hash[key])
        return self.hash[key].next.val   # key对应的值存储在key对应节点的下一个节点中
    def set(self,key,val):
        if key in self.hash:
            self.hash[key].next.val=val
            self.move_to_tail(self.hash[key])
        else:
            node=KeyValue(key,val)
            self.tail.next=node
            self.hash[key]=self.tail
            self.tail=node
            self.size+=1
            if self.size>self.capacity:
                self.hash.pop(self.head.next.key)
                self.head.next=self.head.next.next
                if self.head.next is not None:
                    self.hash[self.head.next.key]=self.head
                self.size-=1

    # def unordered_map


class UglyNumberII():
    def find_kth_ugly_num(self, k):
        # 找到第k个丑数 标准做法 不太能想到 O(n)复杂度
        uglys = []
        uglys.append(1)
        p2, p3, p5 = 0, 0, 0

        for i in range(1, k):
            last_num = uglys[i - 1]
            while uglys[p2] * 2 <= last_num: p2 += 1
            while uglys[p3] * 3 <= last_num: p3 += 1
            while uglys[p5] * 5 <= last_num: p5 += 1
            uglys.append(min(min(uglys[p2] * 2, uglys[p3] * 3), uglys[p5] * 5))
        return uglys[k - 1]

    def new_find_kth_ugly_num(self):
        # Hash Map+Heap(Priority Queue)的方法  O(n * log n)
        # 教程中的代码是java写的，不太好翻译成python 因此这段没写完 ，
        # 紧接着下一个函数是按照该思想写出来的，不知道复杂度还是不是O(n*logn)
        import queue
        Q = queue.Queue()
        inQ = set()
        primes = [None for i in range(3)]

        for i in range(3):
            Q.put(primes[i])
            inQ.add(primes[i])


def new_find_kth_ugly_num(k):
    import heapq
    # 返回第k个丑数  python代码直接用一个集合就可以解决这个问题，
    # 在java中则需要用到HashSet和PriorityQueue
    s = [1]   #  在java中，此处s定义为优先队列，找最小值的操作复杂度为1.

    a = [2, 3, 5]
    min_num = None
    for j in range(k):
        min_num = heapq.heappop(s)
        s.discard(min_num)
        for i in a:
            d = min_num * i
            if d not in s:
                heapq.heappush(s,d)
    return min_num


class TopkLargetstNums():
    # 定义一个数据结构，有添加操作 返回前k个最大数操作，k是固定值，在定义时已确定
    # java中用PriorityQueue代替本代码中的集合，add操作复杂度为O(log n)  topk操作复杂度为O(k*log n)  不明白这个复杂度怎么来的
    def __init__(self, k):
        self.s = set()
        self.k = k

    def add(self, x):
        self.s.add(x)

    def topk(self):
        toplist = []
        for i in range(self.k):
            maxnum = max(self.s)
            self.s.discard(maxnum)
            toplist.append(maxnum)
        for i in toplist:
            self.s.add(i)
        return toplist


class NewTopKLargets():
    # 上题的优化
    # 取前k个最大数，用堆存储k个数，当堆的容量到上限k时，pop出最小的数；由于只保存k个数，空间复杂就是k
    import heapq
    def __init__(self,k=10):
        self.maxsize=k
        self.queue=[]
        self.cur_size=0
    def add(self,item):
        if self.cur_size==self.maxsize:
            minval=heapq.heappop(self.queue)
            if minval<item:
                heapq.heappush(self.queue, item)
            else:
                heapq.heappush(self.queue, minval)
        else:
            heapq.heappush(self.queue,item)
            self.cur_size+=1
    def topk(self):
        return self.queue.sort()



class ListNode():
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next


class MergeKSortedList_v1():
    # merge k个已排序的链表 返回一个排好序的链表

    def merge_k_list(self, node_list):
        if node_list is None or len(node_list) == 0:
            return None
        candidate = set()
        for node in node_list:
            if node is not None:
                candidate.add(node)
        dummy = ListNode()
        tail = dummy
        while len(candidate) != 0:
            min_node = self.find_min_node(candidate)
            head = min_node
            candidate.discard(min_node)  # 教程中用PriorityQueue代替此处的集合，每次pop出一个最小值的节点
            # 这个方法的优势在于用了PriorityQueue的数据结构，能够支持你取最小值，从而简化了算法
            tail.next = head
            if head.next != None:
                candidate.add(head.next)
        return dummy.next

    def find_min_node(self, candidate):
        min_node = ListNode(val=float('-inf'))
        for node in candidate:
            if node.val < min_node.val:
                min_node = node
        return min_node


class MergeKSortedList_v2():
    # 两两归并法
    def merge_k_list(self, node_list):
        if node_list is None or len(node_list) == 0:
            return None
        while len(node_list) > 1:
            new_list = []
            for i in range(0, 2, len(node_list) - 1):
                # 每次取两个node进行merge，并将merge后的新node加入new_list
                merged_list = self.merge(node_list[i], node_list[i + 1])
                new_list.append(merged_list)
            if len(node_list) % 2 == 1:
                new_list.append(node_list[len(node_list) - 1])
            # 用new_list代替原来的list
            node_list = new_list
        return node_list[0]

    def merge(self, node1, node2):
        # merge两个链表
        dummy = ListNode()
        tail = dummy
        while node1 is not None and node2 is not None:
            if node1.val < node2.val:
                tail.next = node1
                node1 = node1.next
            else:
                tail.next = node2
                node2 = node2.next
            if node1 is not None:
                tail.next = node1
            else:
                tail.next = node2
        return dummy.next


class MergeKSortedList_v3():
    # 用分治的思想来做这道题
    def merge_list(self, lists):
        if len(lists) == 0:
            return None
        return self.merge_helper(lists, 0, len(lists) - 1)

    def merge_helper(self, lists, start, end):
        if start == end:
            return lists[start]
        mid = (start + end) // 2
        left = self.merge_helper(lists, start, mid)
        right = self.merge_helper(lists, mid + 1, end)
        return self.merge_two_list(left, right)

    def merge_two_list(self, list1, list2):
        dummy = ListNode()
        tail = dummy
        while list1 is not None and list2 is not None:
            if list1.val < list2:
                tail.next = list1
                tail = list1
                list1 = list1.next
            else:
                tail.next = list2
                tail = list2
                list2 = list2.next
        if list1 is None:
            tail.next = list2
        else:
            tail.next = list1
        return dummy.next


if __name__ == '__main__':

    import queue
    q=queue.Queue()

    import heapq

    heap = [1]
    a = [2, 3, 5]
    for i in range(5):
        minmum = heapq.heappop(heap)
        print(minmum)
        minval = float('inf')
        for j in a:
            if minmum*j not in heap:
                heapq.heappush(heap,minmum*j)

    # return minmum
    pass
