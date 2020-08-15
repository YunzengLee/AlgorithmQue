#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""求二叉树最大深度"""
__author__ = 'Liyunzeng'


# 递归遍历 把当前深度与最大深度比较 打擂台
class Solution(object):
    def __init__(self):
        self.depth = None

    def maxDepth(self, root):
        self.depth = 0
        self.helper(root, 1)
        return self.depth

    def helper(self, node, curDepth):  # 带着参数进去就是遍历
        if node is None:
            return
        if curDepth > self.depth:
            self.depth = curDepth
        self.helper(node.left, curDepth + 1)
        self.helper(node.right, curDepth + 1)


class Solution2(object):
    # 分治法 思想：左子树与右子树的最大值 +1就是全局高度
    def __init__(self):
        pass

    def maxDepth(self, root):
        if root is None:
            return 0
        left = self.maxDepth(root.left)  # 返回结果值就是分治
        right = self.maxDepth(root.right)
        return max(left, right) + 1


class Solution3():
    # 求给定二叉树 返回所有根节点到叶节点的路径,路径是用一个字符串表示的
    def binaryTreePaths(self, root):

        if root is None:
            return []

        paths = []

        if root.left is None and root.right is None:  # 多了对叶子节点的判断，因为万一两个子节点都空，那么会返回两个空列表，
            # 下列for循环操作也就无法进行了
            paths.append(str(root.value))
            return paths

        leftPaths = self.binaryTreePaths(root.left)
        rightPaths = self.binaryTreePaths(root.right)

        for path in leftPaths:
            paths.append(str(root.value) + '>' + path)
        for path in rightPaths:
            paths.append(str(root.value) + '>' + path)

        return paths


class Solution4():
    # 求所有节点之和
    def sumNodes(self, root):
        # 分治
        if root is None:
            return 0
        left = self.sumNodes(root.left)
        right = self.sumNodes(root.right)
        return root.value + left + right

    def sumNodes2(self, root, sum=0):
        # 遍历
        if root is None:
            return sum
        sum += root.value
        left = self.sumNodes2(root.left, sum)
        right = self.sumNodes2(root.right, left)
        return right


class Solution5():
    # LCA问题
    def LowestCommonAncestor(self, root, node1, node2):
        # 1.其中一个节点正是root 则返回root
        # 2.两个节点都在左子树（右子树），返回LCA值
        # 3.两个节点不在左子树（右子树），返回None
        # 4. 左（右）子树仅含一个节点，返回包含的节点
        if root is None or root == node1 or root == node2:
            return root
        # divide
        left = self.LowestCommonAncestor(root.left, node1, node2)
        right = self.LowestCommonAncestor(root.right, node1, node2)

        # conquer
        if left is not None and right is not None:
            # 两个子树都不为空 说明各包含一个节点
            return root
        if left is not None:
            # 说明两个节点都在左子树，返回的就是LCA值，下面同理
            return left
        if right is not None:
            return right
        return None

    def LowestCommonAncestor2(self, root, node1, node2):
        # 如果该树里可能只有一个节点呢?
        # 1.其中一个节点正是root 则返回root
        # 2.两个节点都在左子树（右子树），返回LCA值
        # 3.两个节点不在左子树（右子树），返回None
        # 4. 左（右）子树仅含一个节点，返回包含的节点
        # divide
        is_LCA_l, left = self.LowestCommonAncestor(root.left, node1, node2)
        is_LCA_r, right = self.LowestCommonAncestor(root.right, node1, node2)
        if root is None or root == node1 or root == node2:
            if left is None and right is None:
                return False, root
            else:
                return True, root
        # # divide
        # left = self.LowestCommonAncestor(root.left, node1, node2)
        # right = self.LowestCommonAncestor(root.right, node1, node2)

        # conquer
        if left is not None and right is not None:
            # 两个子树都不为空 说明各包含一个节点
            return True, root
        if left is not None:
            # 说明两个节点都在左子树，返回的就是LCA值，下面同理
            if is_LCA_l:
                return True, left
            else:
                return False, left
        if right is not None:
            if is_LCA_r:
                return True, right
            else:
                return False, right
        return None

    def LowestCommonAncestor3(self, root, node1, node2):
        # 自制
        def helper(root, node1, node2):
            if root is None:
                return 0, None

            leftnum, leftres = helper(root.left, node1, node2)
            rightnum, rightres = helper(root.right, node1, node2)

            sumnum = leftnum + rightnum
            if root == node1 or root == node2:
                return sumnum + 1, root
            else:
                if leftnum == 0 and rightnum == 0:
                    return sumnum, None
                elif (leftnum == 0 and rightnum == 1) or rightnum == 2:
                    return sumnum, rightres
                elif (leftnum == 1 and rightnum == 0) or leftnum == 2:
                    return sumnum, leftres
        num, node = helper(root, node1, node2)
        if num == 2:
            return node
        else:
            return None


class Solution6():
    # 找出二叉树中最长的连续序列的长度  ######分治加遍历
    # 找长度需要分治 找最长需要遍历
    def __init__(self):
        self.maxLength = None

    def maxDep(self, root):
        self.maxLength = 0
        _ = self.helper(root)
        return self.maxLength

    def helper(self, root):
        if root is None:
            return 0
        left_dep = self.helper(root.left)
        right_dep = self.helper(root.right)

        subTreeDep = 1
        if root.left is not None and root.val + 1 == root.left.val:
            subTreeDep = max(subTreeDep, left_dep + 1)
        if root.right is not None and root.val + 1 == root.right.val:
            subTreeDep = max(subTreeDep, right_dep + 1)
        if subTreeDep > self.maxLength:
            self.maxLength = subTreeDep
        return subTreeDep


class Solution6_plus():
    def __init__(self):
        self.max_dep = None

    def get_max_dep(self, root):
        if root is None:
            return None

        self.max_dep = float('-inf')
        self.helper(root)
        return self.max_dep

    def helper(self, node):
        if node is None:
            return None

        # 左右两侧返回的最大长度
        left_leng = self.helper(node.left)
        right_leng = self.helper(node.right)

        cur_leng = 1
        if node.left is not None and node.val + 1 == node.left.val:
            cur_leng = max(left_leng + 1, cur_leng)
        if node.right is not None and node.val + 1 == node.right.val:
            cur_leng = max(right_leng + 1, cur_leng)
        if cur_leng > self.max_dep:
            self.max_dep = cur_leng
        return cur_leng


class Solution7():
    def __init__(self):
        self.target = None

    def getPath_new(self, root, target):
        # 找出所有节点和等于target的路径
        PathListRes = []
        if root is None:
            return PathListRes
        cur_path = []
        cur_path.append(root.val)
        sum = 0
        target = target
        self.helper(root, sum, PathListRes, cur_path, target)
        return PathListRes

    def helper(self, root, sum, PathListRes, cur_path, target):
        # 自上而下  这是遍历法 不是分治法 因为没有返回值，而且带着全局变量
        """

        :param root:
        :param sum: 当前和
        :param PathListRes:全局变量，传入函数中并被不断修改
        :param cur_path: 当前路径
        :param target: 全局变量，目标值，传入函数中并不断用来比较
        :return:
        """
        if root.left is None and root.right is None:
            sum = sum + root.val
            if sum == target:
                PathListRes.append(cur_path)
        if root.left is not None:
            self.helper(root.left, sum + root.val, PathListRes, cur_path.append(root.left.val), target)
            # 这一句里 cur_path列表已经被改变了（加入了左节点的值） 那么在下面的右节点处理时 传入的cur_path有没有包含左节点的值呢？
            #  答： 这里写错了 这应该是java的语法，append函数是不能用在传递参数里运行的，另外append函数确实会改变列表的内容
        if root.right is not None:
            self.helper(root.right, sum + root.val, PathListRes, cur_path.append(root.right.val), target)

    def getPath(self, root):
        # 根节点到任意叶节点为一条路径，找出路径上节点之和等于target的路径
        # 该函数只能得到所有路径，但无法执行与target的比较，因为分治法是从下往上的，子树会返回一个结果，但在没有根节点的子树中，返回的路径肯定不包含根节点，因此无法在执行过程中判断路径之和是否为target
        # 所以该问题适合用遍历法解决，故不是所有问题都适合分治法。
        # 下面代码写的是如何用分治法找出所有路径，是否等于target还需要判断
        if root is None:
            return []
        if root.left is None and root.right is None:
            return [[root.val]]
        # divide
        left_paths = self.getPath(root.left)
        right_paths = self.getPath(root.right)
        # 治
        paths = []
        for path in left_paths:
            paths.append(path.insert(0, root.val))
        for path in right_paths:
            paths.append(path.insert(0, root.val))
        return paths

    def isBST(self, root):
        # 判断是否为二叉查找树
        result = [False, root.val, root.val]
        if root.left is None and root.right is None:
            return True, root.val, root.val
        if root.left is not None:
            is_l, max_l, min_l = self.isBST(root.left)
            if not is_l:
                return False, 0, 0
            if root.val > max_l:
                result[0] = True
                result[2] = min_l
        if root.right is not None:
            is_r, max_r, min_r = self.isBST(root.right)
            if not is_r:
                return False, 0, 0
            if root.val <= min_r:
                result[0] = True
                result[1] = max_r
        return result

    def new_isBST(self, root):
        # 另一种验证是否二叉查找树的方法，与上一种略不同 这一种把root is None 视为TRUE
        if root is None:
            return True, None, None

        # divide
        is_l, max_l, min_l = self.new_isBST(root.left)
        is_r, max_r, min_r = self.new_isBST(root.right)

        #
        if not is_l or not is_r:  # 判断两边是否符合
            return False, None, None
        is_BST = False
        max_val = root.val
        min_val = root.val
        if max_l is not None:
            if root.val > max_l:
                # max_val=root.val
                min_val = min_l
                is_BST = True
        if min_r is not None:
            if root.val <= min_r:
                is_BST = True
                max_val = max_r
            else:
                is_BST = False
        return is_BST, max_val, min_val


class Solution8():
    def get_subTWithMinSum(self, root):
        if root is None:
            return False

        nodeMin = None
        sum = -1000000  # 设一个无线小
        self.helper(root, sum, nodeMin)
        return nodeMin

    def helper(self, root, sum, nodeMin):
        """

        :param root:
        :param sum: 最小子树的和
        :param nodeMin: 最小子树
        :return:
        """
        # 求哪个子二叉树的节点和最小 返回该子节点
        if root is None:
            return 0

        # 分
        left = self.helper(root.left, sum, nodeMin)
        right = self.helper(root.right, sum, nodeMin)

        # 治
        cur_sum = left + right + root.val
        if cur_sum > sum:
            sum = cur_sum
            nodeMin = root
        return cur_sum


class Solution9():
    # 判断一个树是否为平衡二叉树
    def is_BT(self, root):
        if root is None:
            return 'bushishu'
        return self.isBT(root)

    def isBT(self, root):
        if root is None:
            return True, 0

        # fen
        left_isBT, left_high = self.isBT(root.left)
        right_isBT, right_high = self.isBT(root.right)

        # zhi
        if not left_isBT or not right_isBT:
            return False, None
        if abs(left_high - right_high) > 1:
            return False, None
        high = max(left_high, right_high) + 1
        return True, high

    def getSubTreeWithMaxAve(self, root):
        # 求具有最大平均值的子树
        max_ave = None  # 无穷小
        Node = None
        _, _ = self.helper(root, max_ave, Node)
        return Node

    def helper(self, root, max_ave, Node):
        """

        :param root:
        :param max_ave:最大平均值
        :param Node: 具有最大平均值的子树
        :return:
        """
        if root is None:
            return 0, 0  # 返回节点值之和 和 节点数
        left_sum, left_num = self.helper(root.left, max_ave, Node)
        right_sum, right_num = self.helper(root.right, max_ave, Node)

        if (left_sum + right_sum + root.val) // (1 + left_num + right_num) > max_ave:
            max_ave = (left_sum + right_sum + root.val) // (1 + left_num + right_num)
            Node = root
        return left_sum + right_sum + root.val, 1 + left_num + right_num


class Solution10():
    # 求二叉树高度
    def get_Tree_High(self, root):
        # 分治法
        # 求二叉树高度
        if root is None:
            return 0
        # 分
        left = self.get_Tree_High(root.left)
        right = self.get_Tree_High(root.right)
        # 治
        return max(left, right) + 1

    def get_Tree_High_bianli(self, root):
        ### 遍历法
        max_high = 0
        if root is None:
            return 0
        self.helper(root, max_high, 0)
        return max_high

    def helper(self, root, max_high, cur_high):
        cur_high += 1
        if cur_high > max_high:
            max_high = cur_high
        if root.left is not None:
            self.helper(root.left, max_high, cur_high)
        if root.right is not None:
            self.helper(root.right, max_high, cur_high)


class Solution11():
    # 把二叉树按前序变链表
    def flatten(self, root):
        # 遍历法
        if root is None:
            return []
        res = []
        self.helper(root, res)
        return res

    def helper(self, root, res):
        res.append(root.val)
        if root.left is not None:
            self.helper(root.left, res)
        if root.right is not None:
            self.helper(root.right, res)

    def Flatten(self, root):
        # 分治法
        if root is None:
            return []
        left = self.Flatten(root.left)
        right = self.Flatten(root.right)

        res = left + [root.val] + right
        return res


def a1(list1):
    print(list1)


def a2(list2):
    print(list2)


if __name__ == '__main__':
    a = [1, 2, 3]
    # print(a)
    print(a)
    a.append(4)
    a1(a)
    a.pop()
    a2(a)
