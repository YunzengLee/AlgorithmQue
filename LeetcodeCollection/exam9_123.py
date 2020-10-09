def next_pos(matrix,cur_pos):
    dex = [0,0,1,-1]
    dey = [1,-1,0,0]
    x = cur_pos[0]
    y = cur_pos[1]
    res=[]
    for i in range(4):
        newx = x+dex[i]
        newy = y+dey[i]
        if newx<0 or newy<0 or newx>len(matrix) or newy>len(matrix[0]):
            continue
        res.append((newx,newy))
    return res
def shortestPath():
    n = int(input())
    startend = input().split(' ')
    start = (int(startend[0]),int(startend[1]))
    end = (int(startend[2]),int(startend[3]))
    matrix = []
    for _ in range(n):
        matrix.append(input())
    q = [start]
    s=set()
    s.add(start)
    step = 0
    while q:
        step += 1
        size = len(q)
        for i in range(size):
            cur_position = q.pop(0)
            next_positions = next_pos(matrix,cur_position)
            for next_position in next_positions:
                if matrix[next_position[0]][next_position[1]] in '#@':
                    continue
                if next_position in s:
                    continue
                else:
                    if next_position == end:
                        return step
                    s.add(next_position)
                    q.append(next_position)
    return -1