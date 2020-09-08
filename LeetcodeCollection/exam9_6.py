# nAndM = input().split(' ')
# nAndM = list(map(int,nAndM))
# arr = [i+1 for i in range(nAndM[0])]
# operation = input().split(' ')
# operation = list(map(int, operation))
# for i in operation:
#     if i == 1:
#         num = arr.pop(0)
#         arr.append(num)
#     else:
#         idx = 0
#         while idx < nAndM[0]:
#             arr[idx],arr[idx+1] = arr[idx+1],arr[idx]
#             idx+=2
#     # print(arr)
# arr = list(map(str,arr))
# print(' '.join(arr))
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        Hash = {}
        for i in nums:
            Hash[i] = Hash.get(i,0)+1
        sort = sorted(Hash.items(),key = lambda x:x[1],reverse = True)
        return [sort[i][0] for i in range(k)]
if __name__ == '__main__':
    a=Solution()
    res = a.topKFrequent([1,1,1,2,2,3,3,6,8,8,8,8,9,9],3)
    print(res)