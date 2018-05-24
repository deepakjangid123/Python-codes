#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:12:25 2018

@author: abhi
"""

import collections

graph = {0: [1, 2], 1: [2, 0], 2: []}

def bfs(graph, root):
    seen, queue = set([root]), collections.deque([root])
    res = []
    while queue:
        vertex = queue.popleft()
        res.append(vertex)
        for node in graph[vertex]:
            if node not in seen:
                seen.add(node)
                queue.append(node)
    return res

#bfs(graph, 0)
