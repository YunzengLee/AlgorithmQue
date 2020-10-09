class Node:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


nk = input().split(' ')
n = int(nk[0])
k = int(nk[1])
weight = input().split(' ')
weight_map = {}
idx = 1
for i in weight:
    weight_map[idx] = int(i)
    idx += 1
line_map = {}
for i in range(1, n + 1):
    line_map[i] = []
for _ in range(n - 1):
    line = input().split(' ')
    line_map[int(line[0])].append(int(line[1]))
    line_map[int(line[1])].append(int(line[2]))
root = int(input())
s = set(root)
root = Node(val=root)


def dfs(node, node_no, hashmap, s):
    for next_node_no in hashmap[node_no]:
        if next_node_no not in s:
            s.add(next_node_no)
            new_node = Node(val=next_node_no)
            if node.left is None:
                node.left = new_node
            else:
                node.right = new_node
            dfs(new_node, next_node_no, hashmap, s)
