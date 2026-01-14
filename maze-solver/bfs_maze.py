from collections import deque

#1 = path, 0 = wall
maze = [
    [1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

rows = len(maze)
cols = len(maze[0])

# Source and Goal
start = (0, 0)
goal = (4, 4)

# BFS fringe (queue) and expanded list
fringe = deque()
expanded = []

# Parent dictionary to reconstruct path
parent = {}

# Add start node
fringe.append(start)
parent[start] = None

# Possible moves: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

found = False

while fringe:
    current = fringe.popleft()

    if current == goal:
        found = True
        break

    if current not in expanded:
        expanded.append(current)

        for move in directions:
            next_row = current[0] + move[0]
            next_col = current[1] + move[1]
            next_cell = (next_row, next_col)

            if (
                0 <= next_row < rows and
                0 <= next_col < cols and
                maze[next_row][next_col] == 1 and
                next_cell not in expanded and
                next_cell not in fringe
            ):
                fringe.append(next_cell)
                parent[next_cell] = current

# Reconstruct path
path = []
if found:
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    print("Shortest Path using BFS:")
    print(path)
else:
    print("No path found")

print("\nExpanded Nodes:")
print(expanded)

