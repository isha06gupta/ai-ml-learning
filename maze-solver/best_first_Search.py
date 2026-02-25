import heapq

# Manhattan Distance Heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Best First Search
def best_first_search(grid, start, goal):

    rows = len(grid)
    cols = len(grid[0])

    visited = set()
    pq = []

    # push (heuristic value, position)
    heapq.heappush(pq, (heuristic(start, goal), start))

    parent = {}
    visited.add(start)

    while pq:
        h, current = heapq.heappop(pq)

        print("Visiting:", current)

        if current == goal:
            print("Treasure Found!")
            break

        x, y = current

        # possible movements
        neighbors = [
            (x+1, y),
            (x-1, y),
            (x, y+1),
            (x, y-1)
        ]

        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and grid[nx][ny] != 1:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    heapq.heappush(
                        pq,
                        (heuristic((nx, ny), goal), (nx, ny))
                    )

    # reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)

    path.reverse()
    return path


# Example Grid
# 0 = free cell
# 1 = obstacle
grid = [
    [0,0,0,0],
    [0,1,0,1],
    [0,0,0,0],
    [1,0,1,0]
]

start = (0,0)
goal = (3,3)

path = best_first_search(grid, start, goal)
print("Path:", path)
