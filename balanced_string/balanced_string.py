def balanced_string(a_string: str) -> bool:
    """
    :param a_string: A String
    :return: Returns True if string is balanced with respect to parentheses else False
    """

    stack = []

    # Possible opening parentheses
    opening_parentheses = ('(', '{', '[')

    # Possible closing parentheses
    closing_parentheses = (')', '}', ']')

    # mapping of closing parentheses
    parentheses_dict = {')': '(', '}': '{', ']': '['}

    for element in a_string:

        if element in opening_parentheses:
            stack.append(element)

        elif element in closing_parentheses:
            if stack and stack[-1] == parentheses_dict[element]:
                stack.pop()
            else:
                return False

    if stack:
        return False

    return True

#print(balanced_string('((aa}))'))

