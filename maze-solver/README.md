# Maze Solver using BFS and DFS

This project demonstrates solving a grid-based maze using two classical search algorithms:
- Breadth First Search (BFS)
- Depth First Search (DFS)

## Problem Description

The maze is represented as a 2D grid:
- `1` represents a walkable path
- `0` represents a wall

A start (source) cell and an end (goal) cell are defined.
The objective is to find a path from the start to the goal while avoiding walls.

## Algorithms Used

### Breadth First Search (BFS)
- Uses a queue (FIFO)
- Explores the maze level by level
- Guarantees the shortest path in an unweighted grid

### Depth First Search (DFS)
- Uses a stack (LIFO)
- Explores one path deeply before backtracking
- Does not guarantee the shortest path

## Key Concepts Implemented

- Grid-based maze representation
- Valid move generation (up, down, left, right)
- Fringe (queue for BFS, stack for DFS)
- Expanded (visited) list
- Parent tracking for path reconstruction

## Files

- `bfs_maze.py` — Maze solving using BFS
- `dfs_maze.py` — Maze solving using DFS

## Purpose

This project was implemented to understand search strategies and their behavior in solving state-space problems.
