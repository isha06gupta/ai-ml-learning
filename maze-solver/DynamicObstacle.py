#Obstacle may appear while searching.
import heapq
import random

def dynamic_search(grid,start,goal):

    pq=[(0,start)]
    visited=set()

    while pq:
        _,current=heapq.heappop(pq)

        if current==goal:
            print("Goal reached")
            return

        visited.add(current)

        # simulate new obstacle
        rx,ry=random.randint(0,3),random.randint(0,3)
        grid[rx][ry]=1

        x,y=current

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<4 and 0<=ny<4:
                if grid[nx][ny]==0 and (nx,ny) not in visited:
                    h=abs(nx-goal[0])+abs(ny-goal[1])
                    heapq.heappush(pq,(h,(nx,ny)))

grid=[[0]*4 for _ in range(4)]
dynamic_search(grid,(0,0),(3,3))
