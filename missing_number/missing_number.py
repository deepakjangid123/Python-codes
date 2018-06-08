def missing_number(arr1: list, arr2: list) -> int:
    """
    :param arr1: List of numbers
    :param arr2: List of numbers in which we have to find out missing number from arr1
    :return: Returns a single missing element
    """

    x1, x2 = arr1[0], arr2[0]

    # XOR all the elements of array
    for i in range(1, len(arr1)):
        x1 ^= arr1[i]

    for i in range(1, len(arr2)):
        x2 ^= arr2[i]

    return x1^x2


# print(missing_number([1, 2, 6, 8], [2, 8, 1]))