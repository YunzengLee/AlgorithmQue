class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


'''leetcode56 合并区间'''


class Solution_leet56(object):
    '''给出一个区间的集合，请合并所有重叠的区间。
示例 1:

输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
'''

    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals:
            return []
        intervals.sort()  # 将原列表排序
        res = [intervals[0]]
        for i in intervals:
            last = res[-1]
            if i[0] <= last[1]:
                last[1] = max(i[1], last[1])
            else:
                res.append(list(i))
        return res

'''求数组中两个元素的最短距离'''
class Solution_minDis:
    def minDis(self,nums,num1,num2):
        num1_idx = -1
        num2_idx = -1
        mindis = float('inf')
        for idx in range(len(nums)):
            if nums[idx] == num1:
                num1_idx = idx
                if num2_idx>=0:
                    mindis = min(mindis,abs(num1_idx-num2_idx))
            if nums[idx] == num2:
                num2_idx = idx
                if num1_idx>=0:
                    mindis = min(mindis, abs(num1_idx - num2_idx))
        return mindis

'''三个有序数组中取交集'''
def findCommon(array1,array2,array3):
    i=0
    j=0
    k=0
    while i<len(array1) and j<len(array2)and k<len(array3):
        if array1[i]==array2[j]==array3[k]:
            i+=1
            j+=1
            k+=1
            print(array1[i-1])
        elif array1[i]<array2[j]:
            i+=1
        elif array2[j]<array3[k]:
            j+=1
        else:
            k+=1

'''TopK 高频元素'''


class Solution_leet347(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        Hash = {}
        for i in nums:
            Hash[i] = Hash.get(i, 0) + 1
        sort = sorted(Hash.items(), key=lambda x: x[1], reverse=True)
        return [sort[i][0] for i in range(k)]


import functools


def compare(x, y):
    return x - y


functools.cmp_to_key(compare)

## 注意Hash的get方法用法，以及sorted函数用法
'''leetcode 2.两数相加''' '''没有特殊算法 只是对链表的处理'''


class Solution2(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = l1
        tail = None
        tem = 0
        while l1 is not None and l2 is not None:
            val = l1.val + l2.val + tem
            l1.val = val % 10
            tem = val // 10
            tail = l1
            l1 = l1.next
            l2 = l2.next

        if l1 is None and l2 is not None:
            tail.next = l2
            while tem == 1:
                if l2 is None:
                    l2 = ListNode(0)  # ####没注意的地方：l2前一个结点的next是None，l2本来指向None
                    # 这里l2是指向了新节点 这个新节点并没有加到链表的末尾
                    tail.next = l2  # 因此这个tail变量依然是需要的，负责把新节点加入原链表里

                l2.val += tem
                tem = l2.val // 10  # ####没注意的地方：这里一定要先算进位符号tem，再算l2的val 第一次做时顺序颠倒了
                l2.val %= 10
                tail = l2
                l2 = l2.next


        elif l1 is not None and l2 is None:
            tail.next = l1
            while tem == 1:
                if l1 is None:
                    l1 = ListNode(0)
                    tail.next = l1
                l1.val += tem
                tem = l1.val // 10
                l1.val %= 10

                tail = l1
                l1 = l1.next

        else:
            if tem == 1:
                tail.next = ListNode(1)

        return dummy.next


'''leetcode 3. 无重复字符的最长子串'''


# 用双指针遍历的方法时间太长 下面是滑动窗口方法
class Solution3(object):  # ###############
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == '':
            return 0
        hash = set()
        hash.add(s[0])
        startidx = 0
        endidx = 1
        maxlength = 1
        '''aabaa'''
        while endidx < len(s):
            if startidx == endidx:
                hash.add(s[startidx])
                endidx += 1
                continue  # 此句必须要有因为endidx+1之后可能就超出范围了
            if s[endidx] not in hash:
                hash.add(s[endidx])
                maxlength = max(maxlength, endidx - startidx + 1)
                # print(s[startidx,endidx+1])
                # print(startidx,endidx)
                endidx += 1
            else:
                hash.remove(s[startidx])
                startidx += 1
        return maxlength
    def test(self,string):
        s=set()
        right = 0
        left = 0
        maxlen = 0
        while right<len(string):
            if string[right] not in s:
                s.add(string[right])
                right+=1
                maxlen=max(maxlen,right-left)
            else:
                while string[right] in s:
                    s.remove(string[left])
                    left+=1
        return maxlen


'''leetcode 148.排序链表''' '''对链表进行归并排序 时间O(nlogn)空间O(1)'''


#  #归并排序有分治的思想
class Solution148(object):
    def merge(self, head1, head2):
        dummy = ListNode(0)
        tail = dummy
        while head1 is not None and head2 is not None:
            if head1.val < head2.val:
                tail.next = head1
                tail = head1
                head1 = head1.next
            else:
                tail.next = head2
                tail = head2
                head2 = head2.next
        # tail为最后一个节点
        if head1 is None:
            tail.next = head2
        else:
            tail.next = head1
        return dummy.next

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        if head is None or head.next is None:
            return head

        # 一分为二
        slow = head
        fast = head.next
        while fast is not None and fast.next is not None:  # #####找中点这个地方没想起来怎么写
            fast = fast.next.next
            slow = slow.next
        mid = slow.next
        slow.next = None
        # 此时原链表分成了两段 分别以head和mid为头结点的链表
        # 分
        left = self.sortList(head)
        right = self.sortList(mid)
        # 治
        node = self.merge(left, right)
        return node


'''leetcode 4.两个有序数组的中位数'''


class Solution4(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        length = len(nums1) + len(nums2)
        if length % 2 == 0:
            k = length // 2
            print(self.findknum(nums1, 0, nums2, 0, k))
            print(self.findknum(nums1, 0, nums2, 0, k + 1))
            return (self.findknum(nums1, 0, nums2, 0, k) + self.findknum(nums1, 0, nums2, 0, k + 1)) / 2
        else:
            k = length // 2 + 1
            return self.findknum(nums2, 0, nums1, 0, k)

    def findknum(self, nums1, startidx1, nums2, startidx2, k):
        # k可能为奇 为偶
        if startidx1 >= len(nums1):
            return nums2[startidx2 + k - 1]
        if startidx2 >= len(nums2):
            return nums1[startidx1 + k - 1]
        if k == 1:
            return min(nums1[startidx1], nums2[startidx2])

        if startidx1 + k // 2 - 1 >= len(nums1):
            return self.findknum(nums1, startidx1, nums2, startidx2 + k // 2, k - k // 2)
        if startidx2 + k // 2 - 1 >= len(nums2):
            return self.findknum(nums1, startidx1 + k // 2, nums2, startidx2, k - k // 2)
        if nums1[startidx1 + k // 2 - 1] < nums2[startidx2 + k // 2 - 1]:
            return self.findknum(nums1, startidx1 + k // 2, nums2, startidx2, k - k // 2)
        else:
            return self.findknum(nums1, startidx1, nums2, startidx2 + k // 2, k - k // 2)


'''求n以内的素数（质数）'''


# 思路：从2到i遍历， 看能否整除i
# 从2 到sqrt(i)遍历，看能否整除i
# 只遍历奇数因为偶数一定不是质数
# 从2到sqrt（i）遍历，只遍历其中的质数，因为不能分解为质数相乘的数一定是质数。
# （反之 合数一定能分解为若干个质数相乘）
def sushu(n):
    res = []
    for num in range(2, n + 1):
        flag = True
        for su in res:
            if su ** 2 > num:
                break
            if num % su == 0:
                flag = False

        if flag:
            res.append(num)
    return res


'''leetcode103.二叉树锯齿遍历 宽度优先搜索'''
'''用到了栈和队列 栈用来倒序吐出节点 队列用来暂时存储节点 
在栈内节点没有全部吐出之前 新加入的节点先放到队列里 等到栈空了再按顺序入栈'''


class TreeNode():
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


class Stack:
    def __init__(self):
        self.q = []

    def put(self, lel):
        self.q.append(lel)

    def get(self):
        a = self.q.pop()
        return a

    def empty(self):
        return True if len(self.q) == 0 else False

    def size(self):
        return len(self.q)


import queue


class Solution103(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # 先送右 再送左
        q = Stack()
        q1 = queue.Queue()
        res = []
        if root is None:
            return res
        q.put(root)
        while not q.empty():
            # size = q.size()
            level = []
            # for i in range(size):
            while not q.empty():
                node = q.get()
                level.append(node.val)

                if node.left is not None:
                    q1.put(node.left)
                if node.right is not None:
                    q1.put(node.right)
            res.append(list(level))
            while not q1.empty():
                node = q1.get()
                q.put(node)
            # print([node.val for node in q.q])
            if not q.empty():
                level = []
                # size = q.size()
                # for i in range(size):
                while not q.empty():
                    node = q.get()
                    # print(node.val)
                    level.append(node.val)
                    if node.right is not None:
                        q1.put(node.right)

                    if node.left is not None:
                        q1.put(node.left)
                    # print(level)
                while not q1.empty():
                    node = q1.get()
                    q.put(node)
                res.append(list(level))
        return res


class Solution_mianshi32_iii(object):
    # 与上题相同  解法变了 不使用栈进行倒序操作，直接将队列倒序即可
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        if root is None:
            return res
        stack = [root]  # 这个应该是队列 写错名字了
        # bfs一定不用栈
        # 这个题的关键在于 每一次层次遍历的过程中，从队列中读出节点时，根据这些节点的出现顺序就已经决定了
        # 下一层要先放左节点还是右节点，这个顺序用order标识
        # 读入下一层的节点之后，需要反转一下顺序才行，反转可以用【：：-1】 也可以用一个辅助栈。
        order = 0
        while stack:
            size = len(stack)
            subset = []
            for i in range(size):
                node = stack.pop()
                subset.append(node.val)
                if order % 2 == 0:
                    if node.left is not None:
                        stack.insert(0, node.left)
                    if node.right is not None:
                        stack.insert(0, node.right)
                else:
                    if node.right is not None:
                        stack.insert(0, node.right)
                    if node.left is not None:
                        stack.insert(0, node.left)
            order += 1
            res.append(list(subset))
            stack = stack[::-1]
        return res


'''leetcode611.有效三角形的个数'''


class Solution611():
    # 输入一个数组，输出所有三元组，里面的三个数能组成三角形.返回三元组的个数
    # 组成三角形的条件 ： a<=b<=c and a+b>c
    # 注意：若数组为[1,1,1,1]，则可以组成4个答案，即四个数中取其中三个。
    def triangle_count(self, nums):
        nums.sort()
        ans = 0
        for i in range(1, len(nums)):
            left = 0
            right = i - 1
            while left < right:
                if nums[left] + nums[right] > nums[i]:  # ####一开始这里想复杂了
                    ans += right - left
                    right -= 1
                else:
                    left += 1
        return ans


'''leetcode 73 三色分类 将一个仅包含0 1 2 三种数的数组排序  （双指针）'''


class Solution73(object):
    def sortColors(self, nums):  # ##############
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        if nums is None or len(nums) <= 1:
            return nums
        pl = 0
        pr = len(nums) - 1
        i = 0
        while i <= pr:
            if nums[i] == 0:
                tem = nums[i]
                nums[i] = nums[pl]
                nums[pl] = tem
                pl += 1
                i += 1  # ####第一个地方：去掉这句 【0，0，2，1】的情况就不行了 但elif句就没有这个i+1

            elif nums[i] == 2:
                tem = nums[i]
                nums[i] = nums[pr]
                nums[pr] = tem
                pr -= 1  # 从右边换过来的数可能是0，所以需要再处理一次 此时i还不能+1
            else:
                i += 1
        return nums


'''
若改成
        while i <= pr:
            if nums[i] == 0:
                tem = nums[i]
                nums[i] = nums[pl]
                nums[pl] = tem
                pl += 1

            elif nums[i] == 2:
                tem = nums[i]
                nums[i] = nums[pr]
                nums[pr] = tem
                pr -= 1
            
            i += 1
则【1，2，0】的情况就无法处理了
'''

'''leetcode394 字符串解码   考察栈 递归；类似leetcode856'''


# 输入'3[a]2[c2[b]]'输出'aaacbbcbb'
class Solution394(object):
    # 栈解法
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        num = 0
        this_str = ''
        for i in s:
            if i.isdigit():
                num = num * 10 + int(i)
            elif i.isalpha():
                this_str += i
            elif i == '[':
                stack.append((this_str, num))
                num = 0
                this_str = ''
            else:
                last_str, repeat_num = stack.pop()
                this_str = last_str + this_str * repeat_num
        return this_str


'''leetcode121 买卖股票最佳时期 双指针'''


# 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
# 如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
# 注意你不能在买入股票前卖出股票。
# 例 输入: [7,1,5,3,6,4]
# 输出: 5

class Solution121(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if prices is None or len(prices) <= 1:
            return 0
        minprice = prices[0]
        maxprofit = 0  # 这个不要初始化为负无穷，因为不可能
        for i in range(1, len(prices)):
            if prices[i] < minprice:
                minprice = prices[i]
            else:
                maxprofit = max(maxprofit, prices[i] - minprice)
        return maxprofit


'''leetcode 面试题 04.09  二叉搜索树序列  hard级别
 从左向右遍历一个数组，通过不断将其中的元素插入树中可以逐步地生成一棵二叉搜索树。
 给定一个由不同节点组成的二叉树，输出所有可能生成此树的数组。'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution_mianshi_04_09(object):  # hard级别##################
    def BSTSequences(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # #分治法加dfs 找到左子树和右子树的可能序列是分治，生成当前子树的所有序列需要将左右子树的结果进行merge。
        # merge出的结果有多个方案，用dfs求解
        if root is None:
            return [[]]
        res = self.helper(root)
        return res

    def helper(self, node):
        if node is None:
            return None
        if node.left is None and node.right is None:
            return [[node.val]]
        # 分 返回左右子树的所有可能序列
        left_res = self.helper(node.left)
        right_res = self.helper(node.right)
        res = []
        # 治 由左右子树的结果生成当前子树的结果并返回
        if left_res is not None and right_res is not None:
            # 如果左右子树都有结果（子序列的集合）返回，
            # 则需要对每一对子序列（一个来自左集，一个来自右集）进行merge
            # 每一对子序列的merge又会产生多个方案，需要dfs
            for i in left_res:
                for j in right_res:
                    self.merge(i, j, res, node.val)

        elif right_res is None:
            for i in left_res:
                res.append(list([node.val] + i))
        else:
            for i in right_res:
                res.append(list([node.val] + i))
        return res

    def merge(self, left, right, res, num):
        merge_res = []
        self.merge_helper(0, left, 0, right, [], merge_res)
        for i in merge_res:
            res.append(list([num] + i))

    def merge_helper(self, idx_left, left, idx_right, right, subset, merge_res):
        if idx_left == len(left) and idx_right != len(right):
            subset_ = list(subset)
            for i in range(idx_right, len(right)):
                subset_.append(right[i])
            merge_res.append(list(subset_))
            return
        elif idx_left != len(left) and idx_right == len(right):
            subset_ = list(subset)  # 此处subset copy到 subset_ 避免改变subset的内容（如果改变了subset，之后需要pop回原来的状态，不如此处重新定义一个subset_）
            for i in range(idx_left, len(left)):
                subset_.append(left[i])
            merge_res.append(list(subset_))
            return
        else:
            subset.append(left[idx_left])
            self.merge_helper(idx_left + 1, left, idx_right, right, subset, merge_res)
            subset.pop()
            subset.append(right[idx_right])
            self.merge_helper(idx_left, left, idx_right + 1, right, subset, merge_res)
            subset.pop()


'''leetcode501'''
'''给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。
#中序遍历即可
假定 BST 有如下定义：

结点左子树中所含结点的值小于等于当前结点的值
结点右子树中所含结点的值大于等于当前结点的值
左子树和右子树都是二叉搜索树
'''


class Solution501(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 无额外空间 使用中序遍历 4个全局变量 一个记录前一个节点 一个记录最大出现次数 一个记录当前节点出现次数，一个记录结果列表
        res = []
        if not root:
            return res
        prenode = [None]
        maxval = [0]
        self.helper(root, prenode, [0], maxval, res)
        return res

    def helper(self, node, prenode, cursum, maxval, res):
        if not node:
            return
        self.helper(node.left, prenode, cursum, maxval, res)
        # 此处有问题，prenode是指向节点实例的，在node的左节点运行过之后，prenode并没有改变。把prenode做成列表可以解决，但感觉有点怪怪的。或者把prenode做成全局变量
        # cursum和maxval也应该设成全局变量
        if prenode[0]:  # 如果前一个节点存在
            if node.val == prenode[0].val:
                cursum[0] += 1
            else:
                cursum[0] = 1
            if cursum[0] == maxval[0]:
                res.append(node.val)
            if cursum[0] > maxval[0]:
                maxval[0] = cursum[0]

                while res:
                    res.pop()
                # res.clear()  这一句在leetcode上会出错啊 啥情况
                # 绝对不能直接res=[node.val] 因为这样res就指向了一个新列表，上一层的res不会改变 一定要从原列表基础上改变
                res.append(node.val)
        else:
            # 如果前一个节点不存在，说明是第一个节点
            maxval[0] = 1
            cursum[0] = 1
            res.append(node.val)
        prenode[0] = node
        self.helper(node.right, prenode, cursum, maxval, res)

    def findMode1(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 需要遍历所有节点？ 二叉搜索树的条件如何用上？
        # 答：需要遍历所有节点，但可以使空间复杂度为O(1):进行中序遍历，设四个全局变量记录前一个节点，当前节点值出现次数，最大出现次数，结果列表。
        # 遍历的话 O(n)  O(n)
        if not root:
            return None
        dict = {}
        self.helper1(root, dict)
        maxval = float('-inf')
        res = []
        for key in dict:
            if dict[key] > maxval:
                res = [key]
                maxval = dict[key]
            elif dict[key] == maxval:
                res.append(key)
        return res

    def helper1(self, node, dict):
        if not node:
            return
        if node.val in dict:
            dict[node.val] += 1
        else:
            dict[node.val] = 1
        self.helper1(node.left, dict)
        self.helper1(node.right, dict)


'''找出34512这种递增切分数组中的最小值'''


# 二分法即可
#   坑：注意这几种情况 [1,2,3] [3,1,3] [3,1,3,3]
class Solution_JianZhiOffer11():
    def solution(self, num):
        left = 0
        right = len(num) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if num[mid] < num[right]:
                right = mid

            elif num[mid] > num[right]:  # 为什么一定要跟num[right]比较？跟num[left]比较的话无法适用【1，2，3,4】的情况
                left = mid
            else:  # (num[mid]==num[right])
                right -= 1  # 巧妙，#mid指向的值与right指向的值相等，那就把right左移，这样不会漏掉最小值，
                # 又能缩小范围，改变mid值
        return min(num[left], num[right])


'''求个十百每一位上 所有数字之和'''


def sum(m):
    s = 0
    while m:
        s += m % 10
        m = m // 10
    return s


'''leetcode1103 分糖果II'''


class Solution1103():
    def distributeCandies(self, candies, num_people):
        """
        :type candies: int
        :type num_people: int
        :rtype: List[int]
        """
        total_person = int(-0.5 + (2 * candies + 0.25) ** 0.5)  # 解一元二次方程
        times = total_person // num_people
        left_person = total_person % num_people
        left_candies = candies - total_person * (total_person + 1) // 2
        res = [0 for i in range(num_people)]
        for i in range(num_people):
            if i + 1 <= left_person:
                res[i] = (times + 1) * (i + 1) + num_people * (times * (times + 1)) // 2
            else:
                res[i] = (times) * (i + 1) + num_people * (times * (times - 1)) // 2
        res[left_person] += left_candies
        return res


'''leetcode mianshi57-ii  求所有和为target的连续正数序列'''
'''输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
例：target=9  res=[[2,3,4],[4,5]]'''


# #### 采用双指针(本题也是滑动窗口)，
# 滑动窗口本质上也是双指针，但滑动窗口一定是两个指针都在列表最前端开始，移动方向向后
# 什么时候用滑动窗口？（找在一个一维空间中找所有方案，且子方案连续）

class Solution_mianshi_57_ii(object):
    def findContinuousSequence(self, target):
        """
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        left = 1
        right = 2
        cursum = 3
        while left < right:
            if cursum < target:
                right += 1
                cursum += right
            elif cursum > target:
                cursum -= left
                left += 1
            else:
                subset = [i for i in range(left, right + 1)]
                res.append(subset)
                cursum -= left
                left += 1
        return res


"""双指针 滑动窗口"""
'''leetcode443 压缩字符串'''
'''
给定一组字符，使用原地算法将其压缩。
压缩后的长度必须始终小于或等于原数组长度。
数组的每个元素应该是长度为1 的字符（不是 int 整数类型）。
在完成原地修改输入数组后，返回数组的新长度。
'''
'''例：输入['a','b'],不用修改，返回长度2
['a','a','b'],改为['a','2','b'],返回长度3
['a','a'......'a']（包含12个'a'） 改为['a','1','2'],返回长度3
'''


def test(chars):
    if not chars or len(chars) <= 1:
        return len(chars)
    write = 0
    read = 1
    # char = None
    num = 1
    pre = chars[0]
    while read < len(chars):
        if chars[read] == pre:
            num += 1
            read += 1
        else:
            chars[write] = pre
            write += 1
            if num != 1:
                for i in str(num):
                    chars[write] = i
                    write += 1
            num = 1
            pre = chars[read]
            read += 1
    return write


class Solution443(object):
    # 时间O(n) 空间O(1)
    # 双指针，一个读指针往前走，不断读取，一个写指针记录下一个写入的位置
    def compress(self, chars):
        """
        :type chars: List[str]
        :rtype: int
        """
        if len(chars) == 1:
            return 1
        write = 0
        read = 0
        num = 1
        char = chars[read]
        while read < len(chars):
            if read + 1 == len(chars) or chars[read + 1] != char:
                chars[write] = char
                write += 1
                if num != 1:
                    for i in str(num):
                        chars[write] = i
                        write += 1
                read += 1
                num = 1
                if read < len(chars):
                    char = chars[read]
            else:
                read += 1
                num += 1
        return write
        # 一开始的做法，没注意到在原列表上修改，新建了一个列表存结果，没有用双指针，只是用pre
        # 另外一个缺陷就是： 将一个数字从高位到低位依次输出，不要依次除以10，再逆序，只要for i in str(num)即可
        # res=[]
        # pre=None
        # num=0
        # for i in chars:
        #     if pre is not None:
        #         if i ==pre:
        #             num+=1
        #         else:
        #             res.append(pre)
        #             if num!=1:
        #                 stack=[]
        #                 while num>0:
        #                     stack.append(num%10)
        #                     num=num//10

        #                 while len(stack)>0:
        #                     res.append(str(stack.pop()))
        #             pre=i
        #             num=1
        #     else:
        #         pre=i
        #         num+=1
        # res.append(pre)
        # if num!=1:
        #     stack=[]
        #     while num>0:
        #         stack.append(num%10)
        #         num=num//10

        #     while len(stack)>0:
        #         res.append(str(stack.pop()))
        # return len(res)


'''leetcode206 反转链表，一直用的是迭代方法，还有递归方法可以学一下'''


class Solution206(object):
    def reverseList_digui(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head
        new_head = self.reverseList_digui(head.next)
        head.next.next = head
        head.next = None
        return new_head

    def reverseList_diedai(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        cur = head
        while cur is not None:
            next_node = cur.next
            cur.next = pre
            pre = cur
            cur = next_node
        return pre


'''如果是反转第m到第n个的话，没有什么好办法，只能从头过一遍，先找到第m个，反转下面的m-n个，再接上断点'''

'''扩展：链表成对调换：也不知道在leetcode哪个题了'''


class Solution_reverse_2():
    # 递归法
    def reverse_by_2(self, head):
        if head is None or head.next is None:
            return head
        next_node = head.next
        head.next = self.reverse_by_2(next_node.next)
        next_node.next = head
        return next_node

    def reverse_by_k(self, head, k):
        # 仿照上一个函数，每k个一组进行反转，递归方法。
        cur = head
        for i in range(k):
            if cur is None:
                return head
            cur = cur.next
        knext_node = self.reverse_by_k(cur, k)
        pre = knext_node
        cur = head
        for i in range(k):
            next_node = cur.next
            cur.next = pre
            pre = cur
            cur = next_node
        return pre


'''leetcode1013 将数组分成和相等的三个部分'''


# 一个非空整数数组，分成和相等的三个非空部分，能分则返回True 否则返回False
# 一开始有两个地方想偏了，这是道简单题
class Solution1013(object):
    def canThreePartsEqualSum(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        # 一开始想用双指针，找到使和相等的三个部分的分界点，时间复杂度为（n^2）
        # 但是，三等分数组必定每个部分的和为sum/3，所以只要找和为sum/3的分界点即可
        # 第二点：当指针从左开始向右走，并累加，当和为sum/3的时候，指针就是第一分界点，接下来找第二分界点即可，不必考虑下一个第一分界点。不好描述，自己想想为什么
        part = 0
        cur_sum = 0
        for num in A:
            cur_sum += num
            if cur_sum == sum(A) / 3:
                part += 1
                cur_sum = 0
        if part == 3:
            return True
        return False

        # num_sum = sum(A)
        # sum_part = num_sum // 3
        # if num_sum % 3 != 0:
        #     return False
        #
        # point = 0
        # cur_sum = 0
        # while point < len(A) - 2:
        #     cur_sum += A[point]
        #     point += 1
        #     if cur_sum == sum_part:
        #         break
        # if cur_sum != sum_part:
        #     return False
        # while point < len(A) - 1:
        #     cur_sum += A[point]
        #     point += 1
        #     if cur_sum == sum_part * 2:
        #         return True
        # return False


'''leetcode 面试30/155最小栈'''


# 建一个数据结构，栈，添加功能：以O（1）的复杂度返回最小值，此外push pop操作也是O（1）
# tip：#####使用辅助栈

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.stack2 = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if len(self.stack2) == 0:
            self.stack2.append(x)
        else:
            if self.stack2[-1] >= x:
                self.stack2.append(x)

    def pop(self):
        """
        :rtype: None
        """
        val = self.stack.pop()
        if val == self.stack2[-1]:
            self.stack2.pop()
        return val

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def min(self):
        """
        :rtype: int
        """
        return self.stack2[-1]


# 实现最大堆，堆在python中都是用列表实现的
# 最大堆的实现
class MaxHeap:
    def __init__(self, maxSize=None):
        self.maxSize = maxSize
        self.li = [None] * maxSize
        self.count = 0

    def length(self):
        # 求数组的长度
        return self.count

    def show(self):
        if self.count <= 0:
            print('null')
        else:
            print(self.li[: self.count])

    def add(self, value):
        if self.count >= self.maxSize:  # 判断是否数组越界
            raise Exception('full')

        self.li[self.count] = value  # 将新节点增加到最后
        self._shift_up(self.count)  # 递归构建大堆
        self.count += 1

    def _shift_up(self, index):
        # 往大堆中添加元素，并保证根节点是最大的值:
        # 1.增加新的值到最后一个结点，在add实现； 2.与父节点比较，如果比父节点值大，则交换
        if index > 0:
            parent = (index - 1) // 2  # 找到根节点，根据完全二叉树的规律。堆是一个完全二叉树
            if self.li[index] > self.li[parent]:  # 交换结点
                self.li[index], self.li[parent] = self.li[parent], self.li[index]
                self._shift_up(parent)  # 继续递归从底往上判断

    def extract(self):
        # 弹出最大堆的根节点，即最大值
        # 1.删除根结点，将最后一个结点作为更结点 ； 2.判断根结点与左右结点的大小，交换左右结点较大的
        if not self.count:
            raise Exception('null')
        value = self.li[0]
        self.count -= 1
        self.li[0] = self.li[self.count]  # 将最后一个值变为第一个
        self._shift_down(0)
        return value

    def _shift_down(self, index):
        # 1.判断是否有左子节点并左大于根，左大于右；2.判断是否有右子节点，右大于根
        left = 2 * index + 1  # 找到左右子节点的下标，根据完全二叉树的规律。
        right = 2 * index + 2
        largest = index
        # 判断条件

        # 下面2个条件包含了，判断左右结点那个大的情况。如果为3， 4， 5,：
        # 第一个判断条件使得largest = 1，再执行第二个条件，则判断其左结点与右结点的大小
        if left < self.length() and self.li[left] > self.li[largest]:
            largest = left
        if right < self.length() and self.li[right] > self.li[largest]:
            largest = right

        if largest != index:  # 将 两者交换
            self.li[index], self.li[largest] = self.li[largest], self.li[index]
            self._shift_down(largest)


# 最小堆的实现
# 构造最小堆
class MinHeap():
    def __init__(self, maxSize=None):
        self.maxSize = maxSize
        self.array = [None] * maxSize
        self._count = 0

    def length(self):
        return self._count

    def show(self):
        if self._count <= 0:
            print('null')
        print(self.array[: self._count], end=', ')

    def add(self, value):
        # 增加元素
        if self._count >= self.maxSize:
            raise Exception('The array is Full')
        self.array[self._count] = value
        self._shift_up(self._count)
        self._count += 1

    def _shift_up(self, index):
        # 比较结点与根节点的大小， 较小的为根结点
        if index > 0:
            parent = (index - 1) // 2
            if self.array[parent] > self.array[index]:
                self.array[parent], self.array[index] = self.array[index], self.array[parent]
                self._shift_up(parent)

    def extract(self):
        # 获取最小值，并更新数组
        if self._count <= 0:
            raise Exception('The array is Empty')
        value = self.array[0]
        self._count -= 1  # 更新数组的长度
        self.array[0] = self.array[self._count]  # 将最后一个结点放在前面
        self._shift_down(0)

        return value

    def _shift_down(self, index):
        # 此时index 是根结点
        if index < self._count:
            left = 2 * index + 1
            right = 2 * index + 2
            # 判断左右结点是否越界，是否小于根结点，如果是这交换
            if left < self._count and right < self._count and self.array[left] < self.array[index] and self.array[
                left] < self.array[right]:
                self.array[index], self.array[left] = self.array[left], self.array[index]  # 交换得到较小的值
                self._shift_down(left)
            elif left < self._count and right < self._count and self.array[right] < self.array[left] and self.array[
                right] < self.array[index]:
                self.array[right], self.array[index] = self.array[index], self.array[right]
                self._shift_down(right)

            # 特殊情况： 如果只有做叶子结点
            if left < self._count and right > self._count and self.array[left] < self.array[index]:
                self.array[left], self.array[index] = self.array[index], self.array[left]
                self._shift_down(left)


# mi = MinHeap(10)
# print()
# print('-------小顶堆----------')
# for i in num:
#     mi.add(i)
# mi.show()
# print(mi.length())
# for _ in range(len(num)):
#     print(mi.extract(), end=', ')
# print()
# print(mi.length())


'''leetcode343 整数拆分（也是面试14-i剪绳子）
给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。
 返回你可以获得的最大乘积。'''


# 思路，首先想到的dfs，O(n^2) 找出所有可能组合，但是会超时
# 然后想，答案与n的规模有关，而且与前面的值有联系。
# 也就是ans[n]=max( 1*ans[n-1], 2*ans[n-2]....(n-1)*ans[1], 1*(n-1),2*(n-2)...(n-1)*1 )
# 所以用dp动态规划


class Solution343(object):
    def __init__(self):
        self.max_product = 0

    def cuttingRope(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp不超时
        dp = [0 for _ in range(n + 1)]
        dp[1] = 1
        dp[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[-1]
    # 超时的dfs
    #     if n<=3:
    #         return n-1
    #     self.helper(1,n)
    #     return self.max_product

    # def helper(self,cur_product,left_length):
    #     if left_length==0:
    #         self.max_product=max(self.max_product,cur_product)
    #         return
    #     elif left_length<0:
    #         return
    #     else:
    #         for i in range(2,left_length+1):
    #             self.helper(cur_product*i,left_length-i)
    ###邪门解法 又叫贪婪解法，不知道为啥用3这个数
    # x=n%3
    # a=n//3
    # if x==0:
    #     return 3**a
    # elif x==1:
    #     return 3**(a-1)*4
    # else:
    #     return 3**a*2


'''leetcode365 水壶问题'''
'''
有两个容量分别为 x升 和 y升 的水壶以及无限多的水。
请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？
如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。
你允许：
装满任意一个水壶
清空任意一个水壶
从一个水壶向另外一个水壶倒水，直到装满或者倒空'''


class Solution365(object):
    def canMeasureWater365(self, x, y, z):
        """
        :type x: int
        :type y: int
        :type z: int
        :rtype: bool
        """
        # 像个马尔科夫过程，两个壶剩余的水就是当前状态，动作有限。
        # 动作即装满、清空任意水壶，倒水。会导致状态转移。
        # 当状态转移到满足条件的状态：即两个水壶中的水相加为z，就是结果。
        # 关键在于能否找到这个状态
        # 因此使用深度优先搜索，从一个状态出发，分别执行各个动作得到所有下一可能状态。
        # 但是要记录所有已经出现过的状态，否则这个过程不会结束
        self.stack = [(0, 0)]
        self.seen = set()
        # self.seen.add((0, 0))

        while self.stack:
            remain_x, remain_y = self.stack.pop()
            if remain_x + remain_y == z or remain_y == z or remain_x == z:
                return True
            if (remain_x, remain_y) in self.seen:
                continue
            self.seen.add((remain_x, remain_y))

            self.stack.append((0, remain_y))
            self.stack.append((remain_x, 0))
            self.stack.append((x, remain_y))
            self.stack.append((remain_x, y))

            if remain_x >= y - remain_y:
                self.stack.append((remain_x - (y - remain_y), y))
            else:
                self.stack.append((0, remain_x + remain_y))

            if remain_y >= x - remain_x:
                self.stack.append((x, remain_y - (x - remain_x)))
            else:
                self.stack.append((remain_x + remain_y, 0))
        return False


class LRU():
    class Node():
        def __init__(self, key, val):
            self.prev = None
            self.next = None
            self.val = val
            self.key = key

    def __init__(self, capacity):
        self.capacity = capacity
        self.head = Node(-1, -1)
        self.tail = Node(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.h = {}

    def get(self, key):
        if key in self.h:
            current = self.h[key]
            current.prev.next = current.next
            current.next.prev = current.prev
            self.move_to_tail(self.h[key])
            return self.h[key].val
        else:
            return -1

    def set(self, key, val):
        if key in self.h:
            self.h[key].val = val
            current = self.get(key)
        else:
            if len(self.h) == self.capacity:
                self.h.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.next.prev = self.head
            self.h[key] = Node(key, val)
            current = self.h[key]
            self.move_to_tail(current)

    def move_to_tail(self, node):
        node.prev = self.tail.prev
        node.prev.next = node
        node.next = self.tail
        self.tail.prev = node


class LRUv2():
    class Node():
        def __init__(self, key, val):
            self.next = None
            self.key = key
            self.val = val

    def __init__(self, capacity):
        self.capacity = capacity
        self.head = Node(-1, -1)
        self.tail = self.head
        self.h = {}

    def get(self, key):
        if key in self.h:
            node = self.h[key].next
            self.h[key].next = node.next
            self.h[node.next.key] = self.h[key]
            self.move_to_tail(node, key)
            return node.val
        return -1

    def set(self, key, val):
        if self.get(key) != -1:
            self.h[key].next.val = val
        else:
            if len(self.h) == self.capacity:
                self.h.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.h[self.head.next.key] = self.head
            insert = Node(key, val)
            self.move_to_tail(insert, key)

    def move_to_tail(self, node, key):
        self.h[key] = self.tail
        self.tail.next = node
        self.tail = node
        self.tail.next = None


# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
'''leetcode 面试题07，主站105，给出二叉树的前序 中序遍历，重建二叉树'''


class Solution105(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        return self.buildTreeHelper(preorder, 0, len(preorder), inorder, 0, len(inorder))

    def buildTreeHelper(self, preorder, preorder_pre_p, preorder_last_p, inorder, inorder_pre_p, inorder_last_p):
        print(preorder_pre_p, preorder_last_p, inorder_pre_p, inorder_last_p)
        if preorder_pre_p == preorder_last_p:
            # print(preorder_pre_p)
            return TreeNode(preorder[preorder_pre_p])
        elif preorder_pre_p > preorder_last_p:
            return None
        else:
            root_val = preorder[preorder_pre_p]
            root_node = TreeNode(root_val)
            for i in range(inorder_pre_p, inorder_last_p + 1):
                if inorder[i] == root_val:
                    break
            left_node = self.buildTreeHelper(preorder, preorder_pre_p + 1, preorder_pre_p + (i - inorder_pre_p),
                                             inorder, inorder_pre_p, i - 1)
            right_node = self.buildTreeHelper(preorder, preorder_pre_p + (i - inorder_pre_p) + 1, preorder_last_p,
                                              inorder, i + 1, inorder_last_p)
            root_node.left = left_node
            root_node.right = right_node
            return root_node


'''leetcode 面试题33 二叉搜索树的后续遍历序列'''


# 与上题类似
# 给一个列表，判断是否是一个二叉搜索树的后序遍历
class Solution_mianshi_33(object):
    def verifyPostorder(self, postorder):
        """
        :type postorder: List[int]
        :rtype: bool
        """
        return self.verifyHelper(0, len(postorder) - 1, postorder)

    def verifyHelper(self, startidx, endidx, postorder):
        if startidx >= endidx:
            return True
        rootval = postorder[endidx]
        boundary = startidx
        while postorder[boundary] < rootval:  # ##############
            boundary += 1
        i = boundary
        while postorder[i] > rootval:
            i += 1

        return i == endidx and self.verifyHelper(startidx, boundary - 1, postorder) and self.verifyHelper(boundary,
                                                                                                          endidx - 1,
                                                                                                          postorder)


'''leetcode 面试68-i 二叉搜索树中求两个节点的最近公共祖先'''
'''
由于二叉搜索树的性质，可以避免遍历每个节点
两种方法，递归和非递归'''


class Solution_mianshi_68_i:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 递归
        # minval = min(p.val,q.val)
        # maxval = max(p.val,q.val)
        # if root.val >= minval and root.val<=maxval:
        #     return root
        # elif root.val>maxval:
        #     return self.lowestCommonAncestor(root.left,p,q)
        # else:
        #     return self.lowestCommonAncestor(root.right,p,q)

        # 非递归
        while root is not None:
            if root.val < q.val and root.val < p.val:
                root = root.right
            elif root.val > q.val and root.val > p.val:
                root = root.left
            else:
                return root


'''leetcode 945 使数组唯一的最小增量'''
'''
给定整数数组 A，每次 move 操作将会选择任意 A[i]，并将其递增 1。
返回使 A 中的每个值都是唯一的最少操作次数。
例如：输入：[3,2,1,2,1,7]
输出：6
解释：经过 6 次 move 操作，数组将变为 [3, 4, 1, 2, 5, 7]。'''


class Solution945(object):
    def minIncrementForUnique(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        # 超时法1  # 动态规划，从左向右遍历，如果与前面出现的数相同就加一，直到不同。
        # operation_num = 0
        # for i in range(1, len(A)):
        #     while A[i] in A[:i]:
        #         A[i] += 1
        #         operation_num += 1
        # return operation_num

        # 法2
        # 排序，只要当前位置的值小于等于上一个值，就增加到上一个值+1
        # O(nlogn + n)
        A.sort()
        operation_num = 0
        for i in range(1, len(A)):
            if A[i] <= A[i - 1]:
                increase_num = A[i - 1] + 1 - A[i]
                A[i] = A[i - 1] + 1
                operation_num += increase_num
        return operation_num


''''leetcode 面试题45 把数组排成最小的数'''
'''输入: [10,2]
输出: "102"

输入: [3,30,34,5,9]
输出: "3033459" '''


# 关键在于要将数字转成字符串再进行排序，但是'3'与 '30'相比，后者应该放在前面
# 然后默认的排序不是这样，所以要设计一个排序
class Solution_mianshi_45(object):
    def minNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        from functools import cmp_to_key  ###这个函数很关键，可以将对两个数的比较函数变成一个key
        numstr = list(map(str, nums))
        numstr.sort(key=cmp_to_key(lambda x, y: int(x + y) - int(y + x)))
        # 比较'3'和'30'时，x=3 y=30,匿名函数的返回值为负数，说明是小于关系。
        return ''.join(numstr)


'''leetcode 面试题40 topk问题，取一个数组最小的k个数，顺序无所谓'''


# 两种方法：堆；快速排序思想

class Solution_mianshi_40(object):
    def getLeastNumbers(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: List[int]
        """

        # 用堆时间复杂度为  nlogk  空间复杂度为k
        # import heapq
        # res=[]
        # if k==0:
        #     return res
        # for i in range(k):
        #     heapq.heappush(res,-arr[i])

        # for i in range(k,len(arr)):
        #     if -res[0]>arr[i]:
        #         heapq.heappop(res)
        #         heapq.heappush(res,-arr[i])
        # return list(map(lambda x:-x,res))

        # 快速排序  空间复杂度看递归深度，logn-n之间 时间复杂度是n，证明很复杂
        def fastSort(startidx, endidx, arr, k):
            left = startidx
            right = endidx
            key = arr[left]
            while left < right:
                while left < right and arr[right] >= key:
                    right -= 1
                arr[left] = arr[right]
                while left < right and arr[left] <= key:
                    left += 1
                arr[right] = arr[left]
            arr[left] = key
            if left + 1 == k:  ######注意此处：left是个下标，当left+1==k时，表示left指向第k个数
                return arr[:k]
            elif left + 1 < k:
                return fastSort(left + 1, endidx, arr, k)
            else:
                return fastSort(startidx, left - 1, arr, k)

        if k == 0:
            return []
        return fastSort(0, len(arr) - 1, arr, k)


'''leetcode 55 跳跃游戏'''
'''
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个位置。

示例 1:
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
'''


class Solution55:
    def canJump(self, nums: List[int]) -> bool:
        # 贪心算法  不是很理解这种算法的特点。
        farMost = 0
        for i in range(len(nums)):  # 注意：此处i可以等于 len（nums）-1，因为内部还有判断i是否小于最远距离
            if i <= farMost:
                farMost = max(farMost, i + nums[i])
                if farMost >= len(nums) - 1:
                    return True
        return False

    # 深度优先搜索 超时
    def canJumpDFS(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        stack = [0]
        s = set()
        s.add(0)
        while stack:
            cur_idx = stack.pop()
            if cur_idx == len(nums) - 1:
                return True
            if cur_idx < len(nums) and nums[
                cur_idx] != 0:  # cur_idx代表当前所处的位置，是由上一时刻的位置加上步数得到的，步数可能非常大，以至于cur_idx超出范围，所以需要加一个判断
                for i in range(1, nums[cur_idx] + 1):
                    next_idx = cur_idx + i
                    if next_idx not in s:
                        s.add(next_idx)
                        stack.append(next_idx)
        return False


if __name__ == '__main__':
    import heapq

    res = []
    heapq.heappush(res, 1)
    print(res)


    def test(arr, k):
        if len(arr) <= k:
            return arr
        start = 0
        end = len(arr) - 1

        def fastsort(start, end, arr, k):
            left = start
            right = end
            key = arr[start]
            while left < right:
                while left < right and arr[right] >= key:
                    right -= 1
                arr[left] = arr[right]
                while left < right and arr[left] <= key:
                    left += 1
                arr[right] = arr[left]
            arr[left] = key
            if left + 1 == k:
                return arr[:left + 1]
            elif left + 1 < k:
                return fastsort(left + 1, end, arr, k)
            else:
                return fastsort(start, left - 1, arr, k)

        return fastsort(start, end, arr, k)
if __name__ == '__main__':
    s=0x10
    a = 0b10
    c=0o10
    print(a,c,s)