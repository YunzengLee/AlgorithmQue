#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# @param matrix int整型二维数组
# @return int整型一维数组
#
class Solution:
    def SpiralMatrix(self , matrix ):
        # write code here
        res = []
        pos_m = 0
        pos_n = -1
        passed = 0
        m = len(matrix) - 1
        n = len(matrix[0])
        num = (m+1)*n
        while passed < num:
            if passed>=num:
                break
            for _ in range(n):
                pos_n += 1
                res.append(matrix[pos_m][pos_n])
                passed+=1
            n -= 1
            if passed>=num:
                break
            for _ in range(m):
                pos_m += 1
                res.append(matrix[pos_m][pos_n])
                passed+=1
            m -= 1
            if passed>=num:
                break
            for _ in range(n):
                pos_n -= 1
                res.append(matrix[pos_m][pos_n])
                passed+=1
            n -= 1
            if passed>=num:
                break
            for _ in range(m):
                pos_m -= 1
                res.append(matrix[pos_m][pos_n])
                passed+=1
            m -= 1
        return res
if __name__ == '__main__':
    a=Solution()
    a.SpiralMatrix([[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12] ])
    pass
