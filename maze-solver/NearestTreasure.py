#There are multiple treasure present , choose the nearest treasure
import heapq

def heuristic(pos, treasures):
    return min(abs(pos[0]-t[0])+abs(pos[1]-t[1]) for t in treasures)


def multi_treasure(grid,start,treasures):

    pq=[]
    heapq.heappush(pq,(heuristic(start,treasures),start))

    visited=set([start])

    while pq:
        _,current=heapq.heappop(pq)

        if current in treasures:
            print("Treasure found at",current)
            return

        x,y=current

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<len(grid) and 0<=ny<len(grid[0]):
                if grid[nx][ny]==0 and (nx,ny) not in visited:
                    visited.add((nx,ny))
                    heapq.heappush(
                        pq,
                        (heuristic((nx,ny),treasures),(nx,ny))
                    )

grid=[[0]*5 for _ in range(5)]
multi_treasure(grid,(0,0),[(4,4),(2,1)])
