class Solution:
    def numberofprize(self , a , b , c ):
        # write code here
        record = [a,b,c]
        minval = min(record)
        res = minval
        while min(record)>=res:
            print(record)
            res = max(res,min(record))
            minidx = 0
            while record[minidx] != min(record):
                minidx += 1
            for i in range(len(record)):
                if i != minidx and record[i] == max(record):
                    record[i] -= 1
                    break
            for i in range(len(record)):
                if i != minidx and record[i] == max(record):
                    record[i] -= 1
                    break
            record[minidx]+=1
        return res
class Solution2:
    def getHouses(self , t , xa ):
        # write code here
        house_bandary = []
        idx = 0
        while idx < len(xa)-1:
            house_bandary.append(xa[idx]-xa[idx+1])
            house_bandary.append(xa[idx]+xa[idx+1])
            idx += 2
        print(house_bandary)
        house_bandary = house_bandary[1:-1]
        print(house_bandary)
        idx = 0
        res = 2
        while idx<len(house_bandary)-1:
            if house_bandary[idx+1] - house_bandary[idx] > t:
                res+=2
            elif house_bandary[idx+1] - house_bandary[idx] == t:
                res+=1
            idx+=2
        return res
if __name__ == '__main__':
    # a=Solution()
    # res = a.numberofprize(4,4,2)
    # print(res)
    a = Solution2()
    res =a.getHouses(2,[-1,4,5,2])
    print(res)

