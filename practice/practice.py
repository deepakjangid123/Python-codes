import time
from copy import deepcopy
import re
from functools import reduce
from math import sqrt


# Normal integers
a = 4821
print(a)

# Octal literals (base 8)
# A number prefixed by 0o (zero and a lowercase "o" or uppercase "O") will be interpreted as an octal number
a = 0o10
# 8
print(a)

# Hexadecimal literals (base 16)
# Hexadecimal literals have to be prefixed either by "0x" or "0X".
a = 0xA0F
# 2575
print(a)

# Binary literals (base 2)
# Binary literals can easily be written as well. They have to be prefixed by a leading "0", followed by a "b" or "B":
a = 0b101010
# 42
print(a)

# Integers in Python3 can be of unlimited size and there is no "long int" in Python3 anymore.
x = 787366098712738903245678234782358292837498729182728
print(x * x * x)

# Floating-point numbers
x, y = 42.11, 3.1415e-10
print(x, y)

# Complex numbers are written as <real part> + <imaginary part>j
x = 3 + 4j
y = 2 - 3j
z = x + y
print(z)

# "true division" performed by "/"
# "floor division" performed by "//"
print(5 / 3)
print(5 // 3)

# Transform decimal to hex and int to again convert to decimal
print(hex(45))
print(int(0x2d))

# Check if strings have same identity or id(a) == id(b).
# Special characters change identity of strings for eg. id("abhi-abhi") is not equal to id("abhi-abhi"),
# because of hyphen in between
# Try this example in python3 shell
a = "a-!b@"
b = "a-!b@"
# print(a is b)
a = "Abhishek"
b = "Abhishek"
print(a is b)

# Check if strings are equal
a = "Abhishek"
b = "Abhishek"
print(a == b)

# Byte String
x = b"Hallo"
t = str(x)
print(t)
u = t.encode("UTF-8")
print(u)

# A note about efficiency:
# The results of int(10 / 3) and 10 // 3 are equal. But the "//" division is more than two times as fast!
a = time.time()
for x in range(1, 100):
    y = int(100 / x)
print("Time taken", time.time() - a)
a = time.time()
for x in range(1, 100):
    y = 100 // x
print("Time taken", time.time() - a)

# Bitwise negation
print(~3)

# Exponentiation
print(10 ** 3)

# Bitwise XOR
print(6 ^ 3)

# Shift Operators
print(6 << 3)

# If s is a sequential data type, it works like this:
# s[begin: end: step]
# The resulting sequence consists of the following elements:
# s[begin], s[begin + 1 * step], ... s[begin + i * step] for all (begin + i * step) < end.
s = "Python under Linux is great"
print(s[::3])
s = "TPoyrtohnotno  ciosu rtshees  lianr gTeosrto nCtiot yb yi nB oCdaennasdeao"
print(s[::2])
print(s[1::2])
s = "Toronto is the largest City in Canada"
t = "Python courses in Toronto by Bodenseo"
print("".join(["".join(x) for x in zip(s, t)]))

# Different behaviors of `+'
print(3 + 4)
print("HI " + "Abhi")
print(["Hello"] + ["World", "!"])

# List operations
lst = [3, 5, 7]
lst.append(42)
print(lst)
# Pop 0th index
lst.pop(0)
print(lst)
# Pop last index
lst.pop()
print(lst)

# Append list to another list
lst2 = [8, 69]
lst2.append(lst)
print(lst2)
# What if we don't want list to be nested after append operation
lst2 = [8, 69]
lst2.extend(lst)
print(lst2)
lst2.remove(69)
print(lst2)

# Let us look at the time for each of these operations
n = 10000

start_time = time.time()
l = []
for i in range(n):
    l = l + [i * 2]
print(time.time() - start_time)

start_time = time.time()
l = []
for i in range(n):
    l += [i * 2]
print(time.time() - start_time)

start_time = time.time()
l = []
for i in range(n):
    l.append(i * 2)
print(time.time() - start_time)

start_time = time.time()
l = []
for i in range(n):
    l.insert(len(l), i * 2)
print(time.time() - start_time)

# The method "index" can be used to find the position of an element within a list:
# s.index(x[, i[, j]])
# It returns the first index of the value x. A ValueError will be raised, if the value is not present.
# If the optional parameter i is given, the search will start at the index i.
# If j is also given, the search will stop at position j.
colours = ["red", "green", "blue", "green", "yellow"]
print(colours.index("green"))
print(colours.index("green", 2))
print(colours.index("green", 3, 4))
try:
    print(colours.index("green", 5))
except ValueError:
    print("Value is not present in list between specified start and end index")

# Deepcopy
lst1 = ["a", "b", ["c", "d"]]
lst2 = deepcopy(lst1)
print(lst1)
print(lst2)
print("lst1 object is equal to lst2 object:", id(lst1) == id(lst2))

# Dictionaries
# pop
capitals = {'Netherlands': 'Amsterdam', 'Germany': 'Berlin'}
print(capitals.pop("Germany"))

# If key is not present then it will raise KeyError, to prevent that we can provide default value to pop
print(capitals.pop("Switzerland", "Bern"))

# popitem
(country, capital) = capitals.popitem()
print(country, capital)

# Prevent KeyError while accessing a key from a dictionary
locations = {"Toronto": "Ontario", "Vancouver": "British Columbia"}
if "Ottawa" in locations:
    print("Ottawa:", locations["Ottawa"])
if "Toronto" in locations:
    print("Toronto:", locations["Toronto"])

# get
print(locations.get("Toronto"))
print("Setting a default value:", locations.get("Ottawa", "Not Found!"))

# Merging dictionaries
knowledge = {"Frank": {"Perl"}, "Monica": {"C", "C++"}}
knowledge2 = {"Guido": {"Python"}, "Frank": {"Perl", "Python"}}
knowledge.update(knowledge2)
print(knowledge)

# Turn two lists into dictionary
dishes = ["pizza", "sauerkraut", "paella", "hamburger"]
countries = ["Italy", "Germany", "Spain", "USA"]
country_specialities_iterator = zip(countries, dishes)
country_specialities = list(country_specialities_iterator)
print(country_specialities)
country_specialities_dict = dict(country_specialities)
print(country_specialities_dict)
##### Efficient way #####
print(dict(zip(countries, dishes)))

# Behavior of zip: Here z2 will be empty list, because zip returns an iterator
# and it exhausts itself when we compute z1.
l1 = ["a", "b", "c"]
l2 = [1, 2, 3]
c = zip(l1, l2)
z1 = list(c)
print(z1)
z2 = list(c)
print(z2)

# Set
x = {"Hello", "Python"}
#### OR ####
x = set(["Hello", "Python"])
print(x)
x.add("World")
print(x)

# Frozensets: They are nothing but immutable sets
x = frozenset(["Hello", "Python"])
try:
    x.add("World")
except AttributeError:
    print("Frozensets are immutable and don't have attribute 'add'")

# Difference: This method returns the difference of two or more sets as a new set.
x = {"a", "b", "c", "d", "e"}
y = {"b", "c"}
# Both are same
print(x.difference(y))
print(x - y)

# Difference_update: The method difference_update removes all elements of another set from this set.
# x.difference_update(y) is the same as "x = x - y"
x.difference_update(y)
print(x)

# Discard: An element el will be removed from the set, if it is contained in the set.
# If el is not a member of the set, nothing will be done.
x = {"a", "b", "c", "d", "e"}
x.discard("a")
print(x)
x.discard("z")
print(x)

# Remove: works like discard(), but if el is not a member of the set, a KeyError will be raised.
x = {"a", "b", "c", "d", "e"}
x.remove("a")
print(x)
try:
    x.remove("z")
except KeyError:
    print("Key isn't present!")

# Union: This method returns the union of two sets as a new set, i.e. all elements that are in either set.
x = {"a", "b", "c", "d", "e"}
y = {"c", "d", "e", "f", "g"}
# Both are same
print(x.union(y))
print(x | y)

# Intersection: Returns the intersection of the instance set and the set s as a new set.
# In other words: A set with all the elements which are contained in both sets is returned.
x = {"a", "b", "c", "d", "e"}
y = {"c", "d", "e", "f", "g"}
# Both are same
print(x.intersection(y))
print(x & y)

# isdisjoint: This method returns True if two sets have a null intersection.
x = {"a", "b", "c"}
y = {"c", "d", "e"}
print(x.isdisjoint(y))

# issubset: x.issubset(y) returns True, if x is a subset of y.
# "<=" is an abbreviation for "Subset of" and ">=" for "superset of"
# "<" is used to check if a set is a proper subset of a set.
x = {"a", "b", "c", "d", "e"}
y = {"c", "d"}
print(x.issubset(y))

# issuperset: x.issuperset(y) returns True, if x is a superset of y. ">=" is an abbreviation for "issuperset of".
# ">" is used to check if a set is a proper superset of a set.
x = {"a", "b", "c", "d", "e"}
y = {"c", "d"}
print(x.issuperset(y))

# pop: pop() removes and returns an arbitrary set element. The method raises a KeyError if the set is empty
x = {"a", "b", "c", "d", "e"}
print(x.pop())
print(x.pop())

# Regular Expression
data = open("practice.py").read().lower()
words = re.findall(r"\b[\w-]+\b", data)
print("Document length:", len(words), words.count("import"))
# Count word's occurrences
for word in ["import", "a", "b", "x", "y", "str"]:
    print("The word " + word + " occurs " + str(words.count(word)) + " times in the document.")

# Iterator
cities = ["Berlin", "Vienna", "Zurich"]
iterator_obj = iter(cities)
print(iterator_obj)
print(next(iterator_obj))
print(next(iterator_obj))
print(next(iterator_obj))


# The following function 'iterable' will return True, if the object 'obj' is an iterable and False otherwise.
def iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


for element in [34, [4, 5], (4, 5), {"a": 4}, "dfsdf", 4.5]:
    print(element, "is iterable: ", iterable(element))

# print: By default sep=" "
print("Hello", "World!")
print("Hello", "World!", sep=" -> ")
print("Hello consists of {a:2d} letters and (5 / 3) is {b:3.2f}".format(a=5, b=1.666666666666))
print("Hello consists of {0:2d} letters and (5 / 3) is {1:3.2f}".format(5, 1.666666666666))


# An arbitrary number of keyword parameters
def f(**kwargs):
    print(kwargs)


f(de="German", en="English", fr="French")


# Nested functions
def f():
    x = 42

    def g():
        # Change global to nonlocal, if you want to access x defined in f()
        global x
        x = 43

    print("Before calling g: " + str(x))
    print("Calling g now:")
    g()
    print("After calling g: " + str(x))


f()
print(x)


# Decorators
def succ(x):
    return x + 1


successor = succ
# Now we have multiple names for the same function
print(successor(10) == succ(10))

# If we delete one definition then also we will left with the other one
del succ
print(successor(10))


# Functions as parameters
def g():
    print("Hi, it's me 'g'")
    print("Thanks for calling me")


def f(func):
    print("Hi, it's me 'f'")
    print("I will call 'func' now")
    func()
    print("func's real name is " + func.__name__)


f(g)


def polynomial_creator(a, b, c):
    def polynomial(x):
        return a * x ** 2 + b * x + c

    return polynomial


p1 = polynomial_creator(2, 3, -1)
p2 = polynomial_creator(-1, 2, 1)

for x in range(-2, 2, 1):
    print(x, p1(x), p2(x))


# A simple decorator
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return function_wrapper


def foo(x):
    print("Hi, foo has been called with " + str(x))


foo = our_decorator(foo)

foo(42)


# We can also write a decorator with '@'
@our_decorator
def foo(x):
    print("Hi, foo has been called with " + str(x))


foo("HI")


# Checking arguments with a decorator
def argument_test_natural_number(f):
    def helper(arg):
        if type(arg) == int and arg > 0:
            return f(arg)
        else:
            return "Argument is not an integer"

    return helper


@argument_test_natural_number
def sum_of_natural_numbers(num):
    sum = 0
    for i in range(1, num + 1):
        sum += i

    return sum


for i in range(1, 10):
    print(i, sum_of_natural_numbers(i))

print(sum_of_natural_numbers(-1))


# Classes nested functions
# The __call__ method
class A:

    def __init__(self):
        print("An instance of A was initialized")

    def __call__(self, *args, **kwargs):
        print("Arguments are:", args, kwargs)


x = A()
print("now calling the instance:")
x(3, 4, x=11, y=10)
print("Let's call it again:")
x(3, 4, x=11, y=10)


# Using a class as a decorator
class decorator2:

    def __init__(self, f):
        self.f = f

    def __call__(self):
        print("Decorating", self.f.__name__)
        self.f()


@decorator2
def foo():
    print("inside foo()")


foo()


# Memoize function
def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper


@memoize
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


print(fib(42))


# Dir: With the built-in function dir() and the name of the module as an argument,
# you can list all valid attributes and methods for that module.
print(dir(time))


# Regular Expressions
# Named Backreferences
string = "Sun Oct 14 13:47:03 CEST 2012"
expr = r"\b(?P<hours>\d\d):(?P<minutes>\d\d):(?P<seconds>\d\d)\b"
x = re.search(expr, string)
print(x.group('hours') + "," + x.group('minutes') + "," + x.group('seconds'))


# Reduce
# Determining the maximum of a list of numerical values by using reduce
f = lambda a,b: a if (a > b) else b
print(reduce(f, [42, 34, 67, 101, 11]))

# Calculating the sum of the numbers from 1 to 100
print(reduce(lambda x,y: x + y, range(1, 101)))


# List Comprehension
print([(x,y,z) for x in range(1, 30) for y in range(x, 30) for z in range(y, 30) if x**2 + y**2 == z**2])

# Calculation of the prime numbers between 1 and 100 using the sieve of Eratosthenes
noprimes = [j for i in range(2, 8) for j in range(i*2, 100, i)]
primes = [i for i in range(2, 100) if i not in noprimes]
print(primes)

# General way to calculate primes
n = 100
no_primes = [j for i in range(2, int(sqrt(n))) for j in range(i*2, n, i)]
primes = [i for i in range(2, n) if i not in no_primes]
print(primes)

# Exception Handling
try:
    f = open("random-name.txt")
except IOError as e:
    errno, stderror = e.args
    print("I/O error({0}): {1}".format(errno, stderror))