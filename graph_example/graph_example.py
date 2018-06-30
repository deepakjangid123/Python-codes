def find_all_paths(graph: dict, start: int, end: int, path=[]) -> list:

    path = path + [start]

    if start == end:
        return [path]

    if not graph[start]:
        return []

    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)

    return paths


def find_shortest_path_with_almost_same_colored_nodes(all_paths: list, color: list, arr: list) -> int:

    distance = []

    for i in range(len(all_paths)):
        count = 0
        for j in range(len(all_paths[i]) - 1):
            count += arr[all_paths[i][j]][all_paths[i][j + 1]]
        distance.append(count)

    sorted_paths = [x for _, x in sorted(zip(distance, all_paths))]
    distance.sort()

    for i in range(len(sorted_paths)):
        white = 0
        black = 0
        for j in range(len(sorted_paths[i])):
            if color[j] == 0:
                black += 1
            else:
                white += 1
        if abs(white - black) <= 1:
            return distance[i]

    return -1


if __name__ == "__main__":

    N, M = [int(x) for x in input().split()]
    arr = [[0 for x in range(N)] for y in range(N)]
    graph = dict()

    for i in range(N):
        graph[i] = []

    while(M):
        u, v, l = [int(x) for x in input().split()]
        arr[u-1][v-1] = l
        graph[u-1].append(v-1)
        M -= 1

    color = [int(x) for x in input().split()]
    all_paths = find_all_paths(graph, 0, N-1)

    print(find_shortest_path_with_almost_same_colored_nodes(all_paths, color, arr))