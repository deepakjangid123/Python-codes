#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:27:21 2018

@author: abhi
"""

from funcy import project

"""lambda"""
#Small anonynous functions can be created with the lambda keyword
def inc_num(number):
    return lambda x: x + number

fn = inc_num(50)
fn(100) #100 + 50 = 150

pairs = [(10, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[0])

print (lambda x,y: x * y)(3, 4)

"""map"""
def square(x):
    return x**2

squares = map(square, range(10))
squares_using_lambda = map(lambda x: x**2, range(10))

"""filter"""
even_squares = filter(lambda x: x % 2 == 0, squares)

"""reduce"""
dictionary = {'a': 1, 'b': 4}
reduce(lambda x, value:x + value, dictionary.values(), 100)

"""Iterate over a dictionary"""
dict((key, value) for key, value in dictionary.iteritems() if value % 2 == 0)

"""OR"""

foodict = {key: value for key, value in dictionary.items() if value % 2 == 0}

"""Select keys from a dictionary"""
small_dict = project(dictionary, ['a'])

