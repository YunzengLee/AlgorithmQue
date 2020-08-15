'''宽搜算法难题整理'''

'''
leetcode 662 求二叉树最大宽度
关键在于记录二叉树节点的位置信息入队列。
'''


# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution662(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # bfs 将节点入队时保存该节点的位置信息。
        res = 0
        if root is None:
            return res
        queue = [(root, 0)]
        while (queue):
            size = len(queue)
            left_pos = queue[0][1]
            for i in range(size):
                node, pos = queue.pop(0)
                res = max(res, pos - left_pos + 1)
                if node.left is not None:
                    queue.append((node.left, pos * 2))
                if node.right is not None:
                    queue.append((node.right, pos * 2 + 1))
        return res

    # bfs 想法是每次层次遍历时将空节点也存入队列中，计算队列中相邻最远的非空节点的距离。
    # 由于遍历了空节点 甚至遍历了许多不存在的节点，该算法超时。
    #     res = 0
    #     if root is None:
    #         return res
    #     queue = [root]
    #     while any(queue):
    #         res = max(res, self.helper(queue))
    #         size = len(queue)
    #         for i in range(size):
    #             node = queue.pop(0)
    #             if node is None:
    #                 queue.append(None)
    #                 queue.append(None)
    #             else:
    #                 if node.left is not None:
    #                     queue.append(node.left)
    #                 else:
    #                     queue.append(None)
    #                 if node.right is not None:
    #                     queue.append(node.right)
    #                 else:
    #                     queue.append(None)
    #     return res

    # def helper(self, queue):
    #     start = 0
    #     for start in range(len(queue)):
    #         if queue[start] is not None:
    #             break
    #     end = len(queue)-1
    #     for end in range(len(queue)-1, -1 , -1):
    #         if queue[end] is not None:
    #             break
    #     return end - start + 1
if __name__ == '__main__':
    c='c'
    print(c>'a' and c<'d')
    print(2**31-1)
    print(2 ** 31)
    c=str(2**31+2)
    print(str(2**31+2))
    print(int(c))