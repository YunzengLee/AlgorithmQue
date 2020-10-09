def if_heigher(s1,s2):
    symbols0 = '^'
    symbols1 = '*%'
    symbols2 = '+-'
    if s1 in symbols0 and s2 not in symbols0:
        return True
    if s1 in symbols1 and s2 in symbols2:
        return True
    return False
def calculate_helper(symbol,num1,num2):
    if symbol == '^':
        return num1^num2
    if symbol == '/':
        return num1/num2

def calculate(string):
    """
    :param string:
    :return:
    """
    stack = []
    formul = []
    for char in string:
        if char.isdigit():
            formul.append(int(char))
        else:
            if char == '(':
                stack.append(char)
            elif char==')':
                symbol = stack.pop()
                while symbol!='(':
                    formul.append(symbol)
                    symbol = stack.pop()
            else:
                if not stack:
                    stack.append(char)
                else:
                    queue = []
                    peek = stack.pop()
                    while peek!='(':
                        if if_heigher(char, peek):
                            stack.append(peek)
                            stack.append(char)
                            break
                        else:
                            queue.append(peek)
                    stack.append('(')


    while stack:
        formul.append(stack.pop())


    helper_stack = []
    for s in formul:
        if s.isdigit():
            helper_stack.append(s)
        else:
            num1 = helper_stack.pop()
            num2 = helper_stack.pop()
            res = calculate_helper(s,num1,num2)
            helper_stack.append(res)
    return helper_stack.pop()



