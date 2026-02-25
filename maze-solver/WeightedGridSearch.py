#Cells have cost but greedy ignores it.
import heapq

def weighted_grid(grid,start,goal):

    pq=[(0,start)]
    visited=set()

    while pq:
        _,current=heapq.heappop(pq)

        if current==goal:
            print("Reached Goal")
            return

        visited.add(current)
        x,y=current

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<3 and 0<=ny<3:
                if grid[nx][ny]!=-1 and (nx,ny) not in visited:
                    h=abs(nx-goal[0])+abs(ny-goal[1])
                    heapq.heappush(pq,(h,(nx,ny)))

grid=[
[1,3,5],
[2,-1,1],
[4,2,1]
]

weighted_grid(grid,(0,0),(2,2))
