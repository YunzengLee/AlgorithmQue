class Node():
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next


class Solution1():
    def reverse_link(self, head_node):
        # 翻转一个以head_node为头结点的链表
        pre = None
        cur = None
        tem = None
        cur = head_node
        while cur != None:
            tem = cur.next
            cur.next = pre
            pre = cur
            cur = tem
        return pre

    def reverse_link_mn(self, head_node, m, n):
        # 将一个以head_node为头结点的链表从第m个节点到第n个节点翻转
        cur = None
        pre = None
        tem = None
        cur = head_node
        i = 1
        while i < m:
            pre = cur
            cur = cur.next
            i += 1
        # 此时cur指向第m个节点 i=m
        # dummy=Node()
        dummy_ = pre  # m前一个
        dummy = cur   # 第m个

        while i <= n:
            tem = cur.next
            cur.next = pre
            pre = cur
            cur = tem
            i += 1
        # 此时cur指向第n+1个节点 i=n+1
        dummy.next = cur
        dummy_.next=pre
        return head_node  # 这里有问题 如果m=1呢 head_node就不是原来的了  要再写一个if判断句来决定返回值

    def reverse_linklist_in_kgroup(self, head, k):
        # [1,2,3,4,5,6,7]  k=3  return:[3,2,1,6,5,4,7]
        dummy = Link_List_Node(val=0)
        dummy.next = head

        prev = dummy
        while prev != None:
            prev = self.reverseKNodes(prev, k)
        return dummy.next

    def reverseKNodes(self, prev, k):
        if k <= 0:
            return None
        if prev == None:
            return None
        nodek = prev
        node1 = prev.next
        for i in range(k):
            if nodek == None:
                return None
            nodek = nodek.next
        if nodek == None:
            return None
        node_plus = nodek.next
        # 以上几句分别找到翻转前的 第一个节点 第k个节点以及第k+1个节点
        # 找到以后就可以放心执行reverse函数，不用管首尾节点的变化
        self.reverse(prev, prev.next, k)
        # 翻转之后 node1就是这k个节点的最后一个 因此要接上node_plus
        # nodek变成了这k个节点的第一个 与之前的prev接上
        node1.next = node_plus
        prev.next = nodek  # 这句是不是有问题啊  经过self.reverse函数之后prev就不再是这k个节点的前一个了
        return node1

    def reverse(self, prev, curt, k):
        for i in range(k):
            temp = curt.next
            curt.next = prev
            prev = curt
            curt = temp


class Link_List_Node():
    def __init__(self, val=None, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


class CloneLinkList():
    def clone_link_list(self, head):
        # 用dict存储映射关系
        if head == None:
            return None
        # 复制所有节点
        node = head
        dict = {}
        while node != None:
            dict[node] = Link_List_Node(val=node.val)
            node = node.next
        node = head
        while node != None:
            dict[node].next = dict[node.next]
            dict[node].random = dict[node.random]

        return dict[head]

    def clone_link_list_new(self, head):
        # 省去dict映射表的空间  把新节点插入到对应旧节点的next位置上
        if head == None:
            return None
        node = head
        while node != None:
            new_node = Link_List_Node(val=node.val)

            next_source_node = node.next
            node.next = new_node
            new_node.next = next_source_node
            node = next_source_node
        node = head
        while node != None:
            next_node = node.next.next

            node.next.random = node.random.next
            node.next.next = next_node.next

            node = next_node
        return head.next


# 以下题目不会改变链表结构
class LinkListCycle():
    # 给一个链表判断有没有环
    def link_list_cycle(self, head):
        node_fast = head.next
        node_slow = head
        # node=head
        while node_fast != node_slow:
            if node_fast.next or node_fast is None:
                return False
            node_fast = node_fast.next.next
            node_slow = node_slow.next
        return True

    def link_list_cycle_ii(self, head):
        # 判断有没有环 并返回环的入口
        if head is None or head.next is None:
            return None
        fast = head
        slow = head
        while fast != None and fast.next is not None:

            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break
        if fast == None or fast.next is None:
            return None
        # 此时 fast=slow
        # 下面这个用到是数学知识 很难看懂  背下好了
        while head != slow.next:
            head = head.next
            slow = slow.next
        return head

    def if_coexist_in_two_link(self, head1, head2):
        # 判断两个链表有没有交集
        # 从链表1走到尾，接上链表2的头节点，判断有无环结构即可
        node = head1
        while node.next is not None:
            node = node.next
        node.next = head2
        node_fast = head2
        node_slow = node
        while node_fast != node_slow:
            if node_fast.next is None or node_fast is None:
                return False
            node_fast = node_fast.next.next
            node_slow = node_slow.next
        return True


class SortLinkList():
    """链表的排序问题"""

    def mergeTwoLists(self, head1, head2):
        # 归并排序两个链表 虽然归并排序对数组的空间复杂度为O(n) 但在链表中，不需要额外的空间
        dummy = Link_List_Node()
        last_node = dummy
        while head1 is not None and head2 is not None:
            if head1.val < head2.val:
                last_node.next = head1
                head1 = head1.next
            else:
                last_node.next = head2
                head2 = head2.next
            last_node = last_node.next
        if head1 is not None:
            last_node.next = head1
        else:
            last_node.next = head2
        return dummy.next

    def mergeOneList(self, head):
        # 归并排序一个列表

        def findMiddle(head):
            if head.next is None:
                return head

            fast = head.next
            slow = head
            while fast is not None and fast.next is not None:
                fast = fast.next.next
                slow = slow.next
            return slow

        def merge(head1, head2):
            dummy = Link_List_Node()
            tail = dummy
            while head1 is not None and head2 is not None:
                if head1.val < head2.val:
                    tail.next = head1
                    head1 = head1.next
                else:
                    tail.next = head2
                    head2 = head2.next
                tail = tail.next
            if head1 is not None:
                tail.next = head1
            else:
                tail.next = head2
            return dummy.next

        if head is None or head.next is None:
            return head
        mid = findMiddle(head)
        right = self.mergeOneList(mid.next)
        mid.next = None
        left = self.mergeOneList(head)
        return merge(left, right)


# 以下题目为排序数组
class SortArray():
    def mergeTwoSortedArray(self, A, B):
        # 将两个已排序数组归并排序
        if A == None and B == None:
            return None
        result = [0 for i in range(len(A) + len(B))]
        i, j, idx = 0, 0, 0
        while i < len(A) and j < len(B):
            if A[i] < B[j]:
                result[idx] = A[i]
                idx += 1
                i += 1
            else:
                result[idx] = B[j]
                idx += 1
                j += 1
            while i < len(A):
                result[idx] = A[i]
                idx += 1
                i += 1
            while j < len(B):
                result[idx] = B[j]
                idx += 1
                j += 1
            return result

    def MergeSortedArray(self, A, B, m, n):
        # A=[4,5,None,None] m=2 B=[1,2] n=2  return [1,2,4,5]
        # m为A有多少元素 n为B有多少元素
        i = m - 1
        j = n - 1
        idx = m + n - 1
        while i >= 0 and j >= 0:
            if A[i] > B[j]:
                A[idx] = A[i]
                idx -= 1
                i -= 1
            else:
                A[idx] = B[j]
                j -= 1
                idx -= 1
        while j >= 0:
            A[idx] = B[j]
            idx -= 1
            j -= 1


class MedianofTwoSortedArrays():
    # 给两个已排序数组，找出其中点，也就是
    # 长度分别为m n的数组A B 排序后找到第（m+n）/2个的数，如果(m+n)是偶数 就返回 中间两个数的均值

    def normal_method(self, A, B):
        k = None
        m = len(A)
        n = len(B)
        # k代表取第几个数  不是取的数的下标
        if (m + n) % 2 == 1:
            k = (m + n) / 2 + 1
        else:
            k = (m + n) / 2
            k2 = k + 1
        if A is None or len(A) == 0:
            return B[k - 1]
        if B is None or len(B) == 0:
            return A[k - 1]

        i = 0
        j = 0
        idx = 0

        while i < m and j < n:
            if A[i] < B[j]:
                arr_idx = A[i]
                i += 1
                idx += 1
            else:
                arr_idx = B[j]
                j += 1
                idx += 1
            if idx == k:
                return arr_idx
        if i == m:  # A已经用完
            arr_idx = B[j]
            j += 1
            idx += 1
            if idx == k:
                return arr_idx
        else:
            arr_idx = A[i]
            i += 1
            idx += 1
            if idx == k:
                return arr_idx

    def findMedionSortedArrays(self, A, B):
        # 非一般的方法
        length = len(A) + len(B)
        if length % 2 == 1:
            return self.findKth(A, 0, B, 0, length // 2 + 1)
        return (self.findKth(A, 0, B, 0, length // 2) + self.findKth(A, 0, B, 0, length // 2 + 1)) / 2

    def findKth(self, A, A_start, B, B_start, k):
        # 该函数用来找到第k个数
        Maxvalue = float('inf')  # Maxvalue设为无穷大值
        if A_start >= len(A):
            return B[B_start + k - 1]
        if B_start >= len(B):
            return A[A_start + k - 1]
        if k == 1:
            return min(A[A_start], B[B_start])
        A_key = A[A_start + k // 2 - 1] if A_start + k // 2 - 1 < len(A) else Maxvalue
        B_key = B[B_start + k // 2 - 1] if B_start + k // 2 - 1 < len(B) else Maxvalue
        if A_key < B_key:
            return self.findKth(A, A_start + k // 2, B, B_start, k - k // 2)  # 丢掉A的前k//2个数，去找A和B的第k-k//2个数
        else:
            return self.findKth(A, A_start, B, B_start + k // 2, k - k // 2)


import queue

if __name__ == '__main__':
    pass
    # q = queue.Queue()
    # q.put(1)
    # q.put(2)
    # print(q.qsize())
    # print(q.empty())
    # a = q.get()
    # print(a)
    # print(5/2)
    # print(q.qsize())
    q=[1]
    q=q[1:]
    print(q)
