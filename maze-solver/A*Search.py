import heapq

goal = [[1,2,3],
        [4,5,6],
        [7,8,0]]

# -----------------------
# Heuristic 1: Misplaced
# -----------------------
def h1(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count


# -----------------------
# Heuristic 2: Manhattan
# -----------------------
def h2(state):
    distance = 0

    for i in range(3):
        for j in range(3):

            value = state[i][j]

            if value != 0:

                goal_x = (value-1)//3
                goal_y = (value-1)%3

                distance += abs(i-goal_x) + abs(j-goal_y)

    return distance


# -----------------------
# Find empty tile
# -----------------------
def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i,j


# -----------------------
# Generate neighbors
# -----------------------
def neighbors(state):

    x,y = find_zero(state)

    moves = [(1,0),(-1,0),(0,1),(0,-1)]

    result = []

    for dx,dy in moves:

        nx,ny = x+dx , y+dy

        if 0<=nx<3 and 0<=ny<3:

            new_state = [row[:] for row in state]

            new_state[x][y],new_state[nx][ny] = new_state[nx][ny],new_state[x][y]

            result.append(new_state)

    return result


# -----------------------
# A* Search
# -----------------------
def astar(start, heuristic):

    pq = []
    heapq.heappush(pq,(heuristic(start),0,start))

    visited = set()

    nodes = 0

    while pq:

        f,g,state = heapq.heappop(pq)

        nodes += 1

        if state == goal:
            return g,nodes

        visited.add(str(state))

        for n in neighbors(state):

            if str(n) not in visited:

                heapq.heappush(pq,(g+1+heuristic(n),g+1,n))

    return None


# -----------------------
# Start State
# -----------------------

start = [[1,2,3],
         [4,0,6],
         [7,5,8]]


# Run with H1
depth1,nodes1 = astar(start,h1)

# Run with H2
depth2,nodes2 = astar(start,h2)


print("Using H1 (Misplaced Tiles)")
print("Solution Depth:",depth1)
print("Nodes Explored:",nodes1)

print("\nUsing H2 (Manhattan Distance)")
print("Solution Depth:",depth2)
print("Nodes Explored:",nodes2)
