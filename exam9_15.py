def te():
    a = input()
    b = input()
    to_check = ['']
    length = 0
    find = False
    while not find:
        length += 1
        size = len(to_check)
        for _ in range(size):
            s = to_check.pop(0)
            to_check.append(s + '0')
            to_check.append(s + '1')
        for s in to_check:
            if_beauty = True
            if length <= len(a):
                for i in range(len(a) - length+1):
                    if s == a[i:i + length]:
                        if_beauty = False
            if if_beauty and length <= len(b):
                for i in range(len(b) - length+1):
                    if s == b[i:i + length]:
                        if_beauty = False
            if if_beauty:
                print(s)
                return length


print(te())
