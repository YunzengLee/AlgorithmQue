def ifPut(x,y):
    """
    :param x:(1,2,3)
    :param y: (3,4,5)
    :return:
    """
    # print('compare')
    if min(y[0],y[1])>min(x[0],x[1]) and max(y[0],y[1])>max(x[0],x[1]):
        return True
    # print(x,y)
    return False



res = []
while True:
    num = input()
    if not num:
        break
    num = int(num)
    boxes = []
    for i in range(num):
        box = input().split(' ')
        box = list(map(int, box))
        box.sort()
        boxes.append(tuple(box))
        for j in range(2):
            source = list(box)
            ele = source.pop(j)
            source.append(ele)
            boxes.append(tuple(source))
    boxes.sort(key=lambda x:(x[0],x[1]))
    print(boxes)
    dp = [boxes[i][2] for i in range(len(boxes))]
    maxheight = dp[0]
    # print('start')
    for k in range(1,len(dp)):
        for j in range(k):
            # print(k,j,ifPut(boxes[j],boxes[k]))
            if ifPut(boxes[j],boxes[k]):
                dp[k] = max(dp[k],dp[j]+boxes[k][2])
                print('put')
                print(k)
                print(j)
                print('end')
        maxheight = max(maxheight,dp[k])
        print('dp:')
        print(dp)
    res.append(maxheight)
for i in res:
    print(i)

