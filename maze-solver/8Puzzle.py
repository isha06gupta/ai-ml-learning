#8-Puzzle using Best First search
import heapq

goal = ((1,2,3),
        (4,5,6),
        (7,8,0))   # 0 = blank

def heuristic(state):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                gx = (val-1)//3
                gy = (val-1)%3
                dist += abs(i-gx) + abs(j-gy)
    return dist


def get_neighbors(state):
    neighbors = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                x,y = i,j

    moves = [(1,0),(-1,0),(0,1),(0,-1)]

    for dx,dy in moves:
        nx,ny = x+dx,y+dy
        if 0<=nx<3 and 0<=ny<3:
            new = [list(row) for row in state]
            new[x][y],new[nx][ny] = new[nx][ny],new[x][y]
            neighbors.append(tuple(tuple(r) for r in new))

    return neighbors


def best_first_8puzzle(start):
    pq=[]
    heapq.heappush(pq,(heuristic(start),start))
    visited=set()

    while pq:
        _,state=heapq.heappop(pq)

        if state==goal:
            print("Goal Reached")
            return

        visited.add(state)

        for n in get_neighbors(state):
            if n not in visited:
                heapq.heappush(pq,(heuristic(n),n))


start=((1,2,3),(4,0,5),(7,8,6))
best_first_8puzzle(start)
