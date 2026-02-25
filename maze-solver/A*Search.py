import heapq

def astar(grid,start,goal):

    pq=[(0,start)]
    g_cost={start:0}
    parent={}

    while pq:
        _,current=heapq.heappop(pq)

        if current==goal:
            print("Optimal Path Found")
            return

        x,y=current

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy

            if 0<=nx<4 and 0<=ny<4:
                new_g=g_cost[current]+1

                if (nx,ny) not in g_cost or new_g<g_cost[(nx,ny)]:

                    g_cost[(nx,ny)]=new_g
                    h=abs(nx-goal[0])+abs(ny-goal[1])
                    f=new_g+h

                    heapq.heappush(pq,(f,(nx,ny)))
                    parent[(nx,ny)]=current


grid=[[0]*4 for _ in range(4)]
astar(grid,(0,0),(3,3))
