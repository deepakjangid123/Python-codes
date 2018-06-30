You are given a graph G consisting of N nodes and M edges.
Each node of G is either colored in black or white. Also, each edge of G has a particular weight.
Now, you need to find the least expensive path between node 1 and node N, such that difference of the number of black nodes and white nodes on the path is no more than 1.

It is guaranteed  does not consist of multiple edges and self loops.

### Input Format:

The first line contains two space separated integers N and M.
Each of the next M lines contains 3 space separated integers u, v, l which denotes that there is an edge from node u to node v having a weight l.
The next line contains N space separated integers where each integer is either 0 or 1.
If the ith integer is 0 it denotes that ith node is black , otherwise it is white.

### Output Format

Output a single integer denoting the length of the optimal path fulfilling the requirements. Print -1 if there is no such path.

### Sample input
6 6 \
1 2 1 \
2 3 1 \
1 4 2 \
4 3 2 \
3 5 2 \
5 6 3 \
1 1 1 0 0 0

### Sample output
7