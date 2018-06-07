from functools import reduce
import time
import math
import collections


def infixToPostfix(infixexpr):
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = list() # Stack()
    postfixList = []
    tokenList = infixexpr.split()

    for token in tokenList:
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        elif token == '(':
            opStack.append(token)
        elif token == ')':
            topToken = opStack.pop()
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while (len(opStack)) and \
               (prec[opStack[len(opStack) - 1]] >= prec[token]):
                  postfixList.append(opStack.pop())
            opStack.append(token)

    while len(opStack):
        postfixList.append(opStack.pop())
    return " ".join(postfixList)


def factorial_reduce(number: int) -> int:
    """
    :param number: Number for which you want to find factorial
    :return: Factorial of the number
    """
    return reduce(lambda x, y: x * y, range(number, 1, -1))


def find_all_prime_numbers_in_interval(start: int, end: int) -> list:
    """
    :param start: Start of the interval
    :param end: End of the interval
    :return: List of all the prime numbers in the interval
    """
    res = []
    for num in range(start, end + 1):
        # prime numbers are greater than 1
        if num > 1:
            flag = 1 # assume number is prime
            for i in range(2, int(math.sqrt(num)) + 1):
                if not (num % i):
                    flag = 0
                    break
            if flag:
                res.append(num)
    return res


def armstrong_number(number: int, power: int) -> int:
    """
    :param number: Number for which we have to find out armstrong number.
    :param power: Exponential power for each digit.
    :return: Armstrong number, eg. number = abcd, power = n then armstrong number = a*n-times + b*n-times + ... so on.
    """
    res = 0
    while number:
        digit = number % 10
        number = int(number / 10)
        res += math.pow(digit, power)
    return res


def count_number_of_vowels(a_string: str) -> dict:
    """
    :param a_string: Input string
    :return: Returns a dictionary containing vowels and count of those vowels
    """
    a_string = a_string.casefold()
    vowels = ['a', 'e', 'i', 'o', 'u']
    frequencies = dict(collections.Counter(a_string))
    dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
    return dictfilt(frequencies, vowels)


def fibonacci(number: int) -> int:
    """
    :param number: Number for which we want to find out fibonacci number
    :return: Returns fibonacci number
    """
    a, b = 0, 1
    if not number:
        return a
    elif number == 1:
        return b
    else:
        while number > 1:
            temp = b
            b += a
            a = temp
            number -= 1
        return b


def power_of_2(number: int) -> list:
    """
    :param number: Number till which we want power of 2
    :return: Returns a list containing power of 2 in range 1 to number
    """
    return list(map(lambda x: 2**x, range(1, number + 1)))


def divide_by_number(x: int, a_list: list) -> bool:
    """
    :param x: Number to be divided
    :param a_list: List of numbers which will be dividing x
    :return: Returns True if x gets divided by at least one another number than it from the list
    """
    filt = list(filter(lambda a: True if not x % a and x != a else False, a_list))
    if len(filt) >= 1:
        return True
    return False


def divisible_by_another_lambda(a_list: list) -> set:
    """
    :param a_list: List containing numbers
    :return: Returns list of numbers which are divisible by another number in that list
    """
    return set(filter(lambda x: divide_by_number(x, a_list), a_list))


def divisible_by_another(a_list: list) -> set:
    """
    :param a_list: List containing numbers
    :return: Returns list of numbers which are divisible by another number in that list
    """
    res = set()
    for i in a_list:
        for j in a_list:
            if (i != j) and (not i % j):
                res.add(i)
    return res


def lcm(x: int, y: int) -> int:
    """
    :param x: First number
    :param y: Second number
    :return: LCM of both the numbers
    """
    if x > y:
        greater = x
    else:
        greater = y

    while(True):
        if ((greater % x == 0) and (greater % y == 0)):
            return greater
        greater += 1


def gcd(x: int, y: int) -> int:
   """
   :param x: First Integer
   :param y: Second Integer
   :return: Returns gcd of two numbers
   """
   while(y):
       x, y = y, x % y

   return x


def lcm_using_gcd(x: int, y: int) -> int:
    """
    :param x: First integer
    :param y: Second integer
    :return: Returns lcm of both numbers
    """

    return (x*y)//gcd(x, y)


def prime_factors(number: int) -> list:
    """
    :param number: Number for which we need to find out prime factors
    :return: Returns list of prime factors
    """
    res = []
    # Check while number is divided by 2
    while number % 2 == 0:
        res.append(2)
        number /= 2

    # Now check for other prime factors in range from 3 to square root of number at step size 2
    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            res.append(i)
            number /= i

    # Now if number is still greater than 2,
    # which means some prime factor has left to append that also to result
    if number > 2:
        res.append(number)

    return res


def find_number_of_triangles(arr: list) -> int:
    """
    :param arr: List of numbers
    :return: Returns the possible number of triangles can be formed by using these list of numbers.
    """
    # Sort the array and initialize count to 0
    n = len(arr)
    arr.sort()
    number_of_triangles = 0  # type: int # Initialize number of triangles

    # Fix first element. We need to go by n-3 elements as the other two elements will be selected from [i+1...n-1]
    for i in range(0, n - 2):

        # Initialize index of the right most element
        k = i + 2

        # Fix the second element
        for j in range(i + 1, n):

            # Find the right most element which is smaller than
            # the sum of these two elements. Because to form a triangle a + b > c should be satisfied.
            # And we have sorted array so till when we will be getting this condition satisfied
            # we can form that much triangles.
            while k < n and arr[i] + arr[j] > arr[k]:
                k += 1

            # Now total numbers of possible triangles can be (k - j - 1).
            # Because k is one index ahead already in last iteration of while.
            number_of_triangles += k - j - 1
    
    return number_of_triangles


def count_of_all_pairs_sum(arr, n, sum):
    """
    :param arr: List of numbers
    :param n: Length of arr
    :param sum: Sum for which we have to find out pairs in arr
    :return: Returns the count of such possible pairs
    """

    m = [0] * 1000 # This number should be greater than the largest number in arr

    # Store counts of elements in map m
    for i in range(0, n):
        m[arr[i]] += 1

    twice_count = 0

    # Iterate through each element and increment
    # the count (Notice that every pair is
    # counted twice)
    for i in range(0, n):

        twice_count += m[sum - arr[i]]

        # if (arr[i], arr[i]) pair satisfies the
        # condition, then we need to ensure that
        # the count is  decreased by one such
        # that the (arr[i], arr[i]) pair is not
        # considered
        if sum - arr[i] == arr[i]:
            twice_count -= 1

    # return the half of twice_count
    return int(twice_count / 2)


if __name__ == '__main__':
    print(factorial_reduce(13))
    print(find_all_prime_numbers_in_interval(4, 100))
    print(armstrong_number(123, 2))
    print(count_number_of_vowels("Hi there, it is ABHI"))
    print(fibonacci(100))
    print(power_of_2(10))
    print(divisible_by_another([3, 12, 65, 54, 39, 102, 339, 221, 7, 7]))
    print(divisible_by_another_lambda([3, 12, 65, 54, 39, 102, 339, 221, 7, 7]))
    print(lcm(3, 4))
    print(gcd(3, 12))
    print(lcm_using_gcd(3, 4))
    print(prime_factors(300))
    print(find_number_of_triangles([10, 21, 22, 100, 101, 200, 300]))
    print(infixToPostfix("A * B + C * D"))
    print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))
    print(count_of_all_pairs_sum([1, 5, 7, -1, 5], 5, 6))