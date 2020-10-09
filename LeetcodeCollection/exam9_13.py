tuple_num = int(input())
res = []
for _ in range(tuple_num):
    nmk = input().split(' ')
    n = int(nmk[0])
    m = int(nmk[1])
    k = int(nmk[2])
    briges = {}
    for i in range(1, n + 1):
        briges[i] = []
    briges_num = 0
    for _ in range(m):
        inp = input().split(' ')
        cost = int(inp(2))
        if cost > k:
            continue
        b1 = int(inp[0])
        b2 = int(inp[1])
        briges[b1].append(b2)
        briges[b2].append(b1)
        briges_num += 1
    if briges_num < n - 1:
        res.append(False)
        continue
    s = set()
    s.add(1)
    q = [1]
    while q:
        cur_island = q.pop(0)
        for next_island in briges[cur_island]:
            if next_island not in s:
                s.add(next_island)
                q.append(next_island)
    if len(s) < n:
        res.append(False)
    else:
        res.append(True)

for r in res:
    if r:
        print('Yes')
    else:
        print('No')