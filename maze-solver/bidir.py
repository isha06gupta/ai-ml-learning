def bidirectional_bfs(graph, start, goal):
    if start == goal:
        return [start], 0

    queue_start = deque([start])
    queue_goal = deque([goal])

    visited_start = {start}
    visited_goal = {goal}

    parent_start = {start: None}
    parent_goal = {goal: None}

    explored = 0
    meeting_node = None

    while queue_start and queue_goal:

        # Expand from start side
        node = queue_start.popleft()
        explored += 1

        for neighbor in graph[node]:
            if neighbor not in visited_start:
                visited_start.add(neighbor)
                parent_start[neighbor] = node
                queue_start.append(neighbor)

                if neighbor in visited_goal:
                    meeting_node = neighbor
                    return build_bidirectional_path(parent_start, parent_goal, meeting_node), explored

        # Expand from goal side
        node = queue_goal.popleft()
        explored += 1

        for neighbor in graph[node]:
            if neighbor not in visited_goal:
                visited_goal.add(neighbor)
                parent_goal[neighbor] = node
                queue_goal.append(neighbor)

                if neighbor in visited_start:
                    meeting_node = neighbor
                    return build_bidirectional_path(parent_start, parent_goal, meeting_node), explored

    return None, explored
