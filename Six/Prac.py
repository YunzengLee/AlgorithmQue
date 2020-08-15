class LinkNode():
    def __init__(self):
        self.next=None
        self.val=None
def reverse_link(headnode):
    pre=None
    cur=headnode
    while cur is not None:
        next=cur.next
        cur.next=pre
        pre=cur
        cur=next
    return pre
def reverse_mn(headnode,m,n):
    pre=None
    cur=headnode
    i=1
    while i< m:
        pre=cur
        cur=cur.next
        i+=1
    # m节点的前一个
    pre_m=pre
    # 第m个节点
    node_m=cur

    pre=None
    while i <=n:
        next=cur.next
        cur.next=pre
        pre=cur
        cur=next
        i+=1
        #此时cur为第n+1个节点  pre为第n个
    pre_m.next=pre
    node_m.next=cur
    if m!=1:
        return headnode
    else:
        return pre
class LinkNodeS():
    def __init__(self):
        self.random=None
        self.val=None
        self.next=None

def clone_link(head):
    point=head
    dict={}
    while point is not None:
        dict[point]=LinkNodeS()
        point=point.next
    point=head
    while point is not None:
        dict[point].val=point.val
        dict[point].next=dict[point.next]
        dict[point].random=dict[point.random]
        point=point.next
    return dict[head]
def reverse_in_k(head,k):
    num=0
    point=head
    while point is not None:
        num+=1
        point=point.next
    start=1
    end=k
    while end<=num:
        head=reverse_mn(head,start,end)
        start+=k
        end+=k
    return head

def if_cycle(head):
    if head.next is None:
        return False
    fastpoint=head.next
    slowpoint=head
    while head.next is not None and head.next.next is None:
        fastpoint=fastpoint.next.next
        slowpoint=slowpoint.next
        if fastpoint==slowpoint:
            return True
    return False
def is_xiangjiao(node1,node2):
    # point1=node1
    # point2=node2
    while node1.next is not None:
        node1=node1.next
    # 首尾相接
    node1.next=node2
    return if_cycle(node2)

def merge_two_sorted_link(node1,node2):
    dummy=LinkNode()
    tail=dummy
    while node1 is not None and node2 is not None:
        if node2.val<node1.val:
            tail.next=node2
            node2=node2.next
        else:
            tail.next=node1
            node1=node1.next
        tail=tail.next
    if node1 is None:
        tail.next=node2
    if node2 is None:
        tail.next=node1
    return dummy.next

def link_merge(head):
    def find_mid(head):
        if head.next is None:
            return
        fast=head



if __name__=='__main__':
    print(10//10)


    pass

