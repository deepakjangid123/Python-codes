def compress(string: str) -> str:
    """
    :param string: String to be compressed
    :return: Returns a compressed string, ex: 'AAAaaBb' will be compressed as 'A3a2B1b1'
    """

    # Initializing result
    res = ""

    # Check if string is empty
    if not string:
        return res

    # Default count for each character
    count = 1

    i = 0

    for i in range(1, len(string)):
        if string[i - 1] == string[i]:
            count += 1
        else:
            res += string[i - 1] + str(count)
            count = 1

    res += string[i] + str(count)

    return res

# print(compress("AabBBC"))
