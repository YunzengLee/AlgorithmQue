def find1(s, e):
    i = 1
    find = False
    length = 0
    while True:
        num1 = s + i
        num2 = s - i
        if e % num1 == 0:
            length = i
            return (length, num1)
        if num2 >= 1 and e % num2 == 0:
            length = i
            return (length, num2)
        i += 1


SandE = input().split(' ')
s = int(SandE[0])
e = int(SandE[1])
if s == e:
    print(0)
    print(1)
    print(s)
elif s > e:
    print(s - e)
    print(s - e + 1)
    print(' '.join([str(i) for i in range(e, s - 1, -1)]))
else:
    if e % s == 0:
        print(0)
        print(2)
        print(str(s) + ' ' + str(e))
    else:
        source_s = s
        res = [str(s)]
        while s < e:
            s = s + source_s
        # s大于e
        length, num = find1(source_s, e)
        minlength = min(length, abs(s - e), abs(s - source_s - e))
        if abs(s - e) == minlength:
            print(abs(s - e))
            while s != e:
                res.append(str(s))
                s -= 1
            res.append(str(s))
            print(len(res))
            print(' '.join(res))
        elif abs(s - source_s - e) == minlength:
            print(abs(s - source_s - e))
            s = s - source_s
            if str(s) != res[0]:
                res.append(str(s))
            s += 1
            while s != e:
                res.append(str(s))
                s+=1
            res.append(str(s))
            print(len(res))
            print(' '.join(res))
        else:
            print(length)
            if num > source_s:
                for i in range(source_s + 1, num + 1):
                    res.append(str(i))
                if num != e:
                    res.append(str(e))
                print(len(res))
                print(' '.join(res))
            else:
                for i in range(source_s - 1, num - 1, -1):
                    res.append(str(i))
                if num != e:
                    res.append(str(e))
                print(len(res))
                print(' '.join(res))
if __name__=='__main__':
    pass