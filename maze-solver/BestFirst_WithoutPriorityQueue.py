#Manual selection of minimum heuristic.
def best_first_no_heap(grid,start,goal):

    open_list=[start]
    visited=set()

    while open_list:

        current=min(
            open_list,
            key=lambda x:abs(x[0]-goal[0])+abs(x[1]-goal[1])
        )

        open_list.remove(current)

        if current==goal:
            print("Goal Found")
            return

        visited.add(current)
        x,y=current

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<4 and 0<=ny<4:
                if (nx,ny) not in visited:
                    open_list.append((nx,ny))

grid=[[0]*4 for _ in range(4)]
best_first_no_heap(grid,(0,0),(3,3))
