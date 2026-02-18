from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# City Map (Graph Representation)
# -------------------------------

graph = {
    'A': ['B', 'D'],
    'B': ['A', 'C', 'E'],
    'C': ['B', 'F'],
    'D': ['A', 'E', 'G'],
    'E': ['B', 'D', 'F'],
    'F': ['C', 'E', 'I'],
    'G': ['D', 'H'],
    'H': ['G', 'I'],
    'I': ['F', 'H']
}

# --------------------------------
# Helper Function to Build Path
# --------------------------------

def build_path(parent_start, parent_goal, meeting):
    path_start = []
    node = meeting
    while node is not None:
        path_start.append(node)
        node = parent_start[node]
    path_start.reverse()

    path_goal = []
    node = parent_goal[meeting]
    while node is not None:
        path_goal.append(node)
        node = parent_goal[node]

    return path_start + path_goal


# --------------------------------
# Bi-Directional BFS Function
# --------------------------------

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

    while queue_start and queue_goal:

        # ---- Expand from start side ----
        current_start = queue_start.popleft()
        explored += 1

        for neighbor in graph[current_start]:
            if neighbor not in visited_start:
                visited_start.add(neighbor)
                parent_start[neighbor] = current_start
                queue_start.append(neighbor)

                if neighbor in visited_goal:
                    path = build_path(parent_start, parent_goal, neighbor)
                    return path, explored

        # ---- Expand from goal side ----
        current_goal = queue_goal.popleft()
        explored += 1

        for neighbor in graph[current_goal]:
            if neighbor not in visited_goal:
                visited_goal.add(neighbor)
                parent_goal[neighbor] = current_goal
                queue_goal.append(neighbor)

                if neighbor in visited_start:
                    path = build_path(parent_start, parent_goal, neighbor)
                    return path, explored

    return None, explored


# --------------------------------
# Run the Search
# --------------------------------

start = 'A'
goal = 'I'

path, explored_nodes = bidirectional_bfs(graph, start, goal)

print("Shortest Path using Bi-Directional BFS:")
print(path)
print("Number of Nodes Explored:", explored_nodes)


# --------------------------------
# Visualization using networkx
# --------------------------------

G = nx.Graph()

for node in graph:
    for neighbor in graph[node]:
        G.add_edge(node, neighbor)

pos = nx.spring_layout(G)

plt.figure(figsize=(8, 6))

# Draw all nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)

# Highlight path if found
if path:
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red')
    edges_in_path = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='red', width=3)

plt.title("City Route Finder using Bi-Directional BFS")
plt.show()
