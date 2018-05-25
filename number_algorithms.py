from functools import reduce
import time
import math
import collections


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