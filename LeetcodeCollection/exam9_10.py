class Solution:
    def lastRemaining(self , n , m ):
        # write code here
        persons = [i for i in range(n)]
        startidx = 0
        while len(persons) != 1:
            choosed_idx = (startidx + m - 1) % len(persons)
            persons.pop(choosed_idx)
            startidx = choosed_idx
        return persons[0]

class Solution1:
    def make_cancellation(self , content , bomb ):
        # write code here
        res = ''
        i = 0
        while i < len(content):
            if i+1<len(content) and content[i] == content[i+1]:
                if content[i] != bomb:
                    i += 2
                else:
                    if res:
                        res = res[:-1]
                    i += 3
            else:
                res = res + content[i]
                i+=1
            print(i,res)
if __name__ =='__main__':
    # a=Solution1()
    # res=a.make_cancellation('112','2')
    # print(res)
    a = [1,2,3]
    b = map(lambda x:x+1, range(6))
    for i in a:
        if i in b:
            print(i)
    a=[1,2,4]
    b=[3,4,5]
    a=a+b
    print(a)
    c=[5,6,7]
    c.extend(b)
    print(c)
    pass