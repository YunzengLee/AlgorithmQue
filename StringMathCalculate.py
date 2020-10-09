"""
常见题型：给一个字符串数学表达式，求计算结果。
答：先得到后缀表达式，再对后缀表达式做处理
"""
def if_heigher(s1,s2):
    # 比较运算符优先级，若s1优先于s2返回True
    ss1 = '+-'
    ss2 = '*/'
    if s1 in ss2 and s2 in ss1:
        return True
    return False
def calculate_helper(s,num1,num2):
    if s=='+':
        return num1+num2
    if s=='-':
        return num1-num2
    if s=='*':
        return num1*num2
    if s=='/':
        return num1//num2

def calculate(string):
    stack = []  # 辅助栈
    formula = []  # 后缀表达式
    for i in string:  # 遍历字符串
        if i.isdigit():  # 如果是数字则直接进入后缀表达式
            formula.append(i)
            continue
        if i=='(':  # 左括号则入栈
            stack.append(i)
        elif i == ')':  # 右括号则将栈顶的元素依次pop出来进入后缀表达式，直到遇见左括号才停止。
            peek = stack.pop()
            while peek != '(':
                formula.append(peek)
                peek = stack.pop()
        else:  # 如果i是一个运算符
            if not stack:  # 栈为空则直接放入
                stack.append(i)
                continue
            peek = stack.pop()
            if peek=='(' or if_heigher(i,peek):  # 若栈顶为左括号或者当前运算符的优先级大于栈顶元素，则直接入栈
                # （也就是说，这个栈只能存放优先级比栈顶元素更高的运算符）
                stack.append(peek)
                stack.append(i)
            else:
                while peek !='(' and not if_heigher(i,peek):  # 如果栈顶元素优先级大等于当前运算符（也就是当前运算符并不绝对优先于栈顶元素）
                                                              #  那就依次pop出栈顶上优先级大于等于当前运算符的元素，并加入后缀表达式，
                                                              # 直到栈为空、或栈顶元素优先级小于当前运算符、或遇到左括号，才停止
                    formula.append(peek)
                    if stack:
                        peek = stack.pop()
                    else:
                        peek = None
                        break
                if peek:
                    stack.append(peek)
                stack.append(i)
    while stack:  # 结束字符串遍历后，将栈内的元素依次放入后缀表达式
        formula.append(stack.pop())
    print(formula)
    print('###')
    # 计算后缀表达式，还是要利用一个辅助栈
    stack_helper=[]
    for i in formula:  # 遍历后缀表达式
        if i.isdigit(): # 遇到数字则入栈
            stack_helper.append(int(i))
        else:  # 遇到运算符则pop出栈顶两个数字进行计算，结果再入栈
            num2 = stack_helper.pop()
            num1 = stack_helper.pop()
            res = calculate_helper(i,num1,num2)
            stack_helper.append(res)
    # 最后辅助栈只会剩余一个元素，就是最终结果
    return stack_helper[0]
if __name__=='__main__':
    pass
    print('')
    # a=[1,23]
    # b=a
    # b.extend([1,2,3])
    # print(a,b)
    # with open('./Java/MyLruCache.java',encoding='utf8') as f:
    #     a=f.readline()
    # #     # print(a)
    #     while(a):
    #         print(a.strip())
    #         a=f.readline()


# s=0x10
# a = 0b10
# c=0o10
# print(a,c,s)
# res=calculate('1*2-2*(3+4)') # 后缀表达式为 ['1', '2', '+', '2', '3', '4', '+', '*', '-']
# print(res)
