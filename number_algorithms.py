from functools import reduce
import time
import math


def factorial_reduce(number):
    """
    :param number: Number for which you want to find factorial
    :return: Factorial of the number
    """
    return reduce(lambda x, y: x * y, range(number, 1, -1))


def find_all_prime_numbers_in_interval(start, end):
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


def armstrong_number(number, power):
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


if __name__ == '__main__':
    print(factorial_reduce(13))
    print(find_all_prime_numbers_in_interval(4, 100))
    print(armstrong_number(123, 2))