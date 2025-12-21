---
title: Chapter 9 - Motion Planning and Navigation
sidebar_position: 3
---

# Chapter 9: Motion Planning and Navigation

## Learning Goals

- Master path planning algorithms
- Understand navigation in dynamic environments
- Learn obstacle avoidance techniques
- Implement A*, Dijkstra, and RRT path planning
- Navigate robots in complex environments
- Handle dynamic obstacles and replanning

## Introduction to Motion Planning

Motion planning is the process of finding a collision-free path from a start configuration to a goal configuration while satisfying various constraints. It's a fundamental capability for autonomous robots that need to navigate in complex environments.

### Motion Planning Components

A complete motion planning system consists of:

1. **Configuration Space (C-space)**: The space of all possible robot configurations
2. **Planning Algorithm**: Method for finding a path through C-space
3. **Collision Detection**: System for detecting obstacles
4. **Path Optimization**: Techniques for smoothing and improving paths
5. **Trajectory Generation**: Conversion of geometric path to timed trajectory

## Configuration Space and Representation

### Configuration Space Concepts

The configuration space (C-space) represents all possible configurations of a robot. For a robot with n degrees of freedom, the C-space is an n-dimensional space.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class ConfigurationSpace:
    def __init__(self, bounds, obstacles=None):
        """
        Initialize configuration space
        bounds: List of (min, max) for each dimension
        obstacles: List of obstacle shapes [(center, shape_params), ...]
        """
        self.bounds = bounds
        self.obstacles = obstacles if obstacles else []
        self.dimensions = len(bounds)

    def is_valid_configuration(self, config):
        """
        Check if configuration is collision-free
        config: Array representing robot configuration
        """
        # Check bounds
        for i, (min_val, max_val) in enumerate(self.bounds):
            if not (min_val <= config[i] <= max_val):
                return False

        # Check obstacles
        for obstacle in self.obstacles:
            center, shape_type, params = obstacle
            if self._check_collision(config, center, shape_type, params):
                return False

        return True

    def _check_collision(self, config, center, shape_type, params):
        """Check collision with obstacle"""
        if shape_type == 'circle':
            radius = params[0]
            distance = np.linalg.norm(np.array(config) - np.array(center))
            return distance < radius
        elif shape_type == 'rectangle':
            width, height = params
            half_width, half_height = width/2, height/2
            rel_pos = np.array(config) - np.array(center)
            return (abs(rel_pos[0]) < half_width and abs(rel_pos[1]) < half_height)
        else:
            return False  # Unknown shape

    def sample_free_space(self):
        """Sample a random configuration in free space"""
        for _ in range(1000):  # Try up to 1000 times
            config = []
            for min_val, max_val in self.bounds:
                config.append(np.random.uniform(min_val, max_val))

            if self.is_valid_configuration(config):
                return np.array(config)

        # If random sampling fails, return None or use grid sampling
        return None

    def get_neighbors(self, config, step_size=0.1):
        """Get neighboring configurations"""
        neighbors = []
        for i in range(self.dimensions):
            for direction in [-1, 1]:
                neighbor = config.copy()
                neighbor[i] += direction * step_size

                if self.is_valid_configuration(neighbor):
                    neighbors.append(neighbor)

        return neighbors


# Example usage
def main():
    # Define configuration space for 2D point robot
    bounds = [(-5, 5), (-5, 5)]  # x and y bounds
    obstacles = [
        ((0, 0), 'circle', (1.0,)),  # Circle at origin with radius 1
        ((2, 2), 'rectangle', (1.0, 1.0)),  # Rectangle at (2,2) with width=1, height=1
    ]

    cspace = ConfigurationSpace(bounds, obstacles)

    # Sample valid configurations
    valid_configs = []
    for _ in range(100):
        config = cspace.sample_free_space()
        if config is not None:
            valid_configs.append(config)

    # Visualize configuration space
    if valid_configs:
        configs = np.array(valid_configs)
        plt.figure(figsize=(10, 8))
        plt.scatter(configs[:, 0], configs[:, 1], alpha=0.6, s=10)

        # Draw obstacles
        circle = plt.Circle((0, 0), 1, color='red', alpha=0.3, label='Circular Obstacle')
        rect = plt.Rectangle((1.5, 1.5), 1, 1, color='red', alpha=0.3, label='Rectangular Obstacle')
        plt.gca().add_patch(circle)
        plt.gca().add_patch(rect)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title('Configuration Space Sampling')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    main()
```

### Discretization and Graph Representation

Motion planning algorithms often discretize the configuration space into a graph:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import heapq


class GridBasedSpace:
    def __init__(self, width, height, resolution=1.0):
        """
        Initialize grid-based configuration space
        width, height: Grid dimensions
        resolution: Size of each grid cell
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((int(height/resolution), int(width/resolution)))  # 0 = free, 1 = obstacle
        self.obstacles = set()

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x + self.width/2) / self.resolution)
        grid_y = int((y + self.height/2) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.resolution - self.width/2
        y = grid_y * self.resolution - self.height/2
        return x, y

    def is_valid_cell(self, grid_x, grid_y):
        """Check if grid cell is valid (not out of bounds or obstacle)"""
        if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
            return self.grid[grid_y, grid_x] == 0
        return False

    def get_neighbors(self, grid_x, grid_y):
        """Get valid neighboring cells (8-connectivity)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = grid_x + dx, grid_y + dy
                if self.is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def add_obstacle(self, x, y, width=1, height=1):
        """Add obstacle to grid"""
        grid_x, grid_y = self.world_to_grid(x, y)
        grid_w = int(width / self.resolution)
        grid_h = int(height / self.resolution)

        for i in range(max(0, grid_x - grid_w//2), min(self.grid.shape[1], grid_x + grid_w//2 + 1)):
            for j in range(max(0, grid_y - grid_h//2), min(self.grid.shape[0], grid_y + grid_h//2 + 1)):
                self.grid[j, i] = 1
                self.obstacles.add((i, j))


class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = {}  # vertex -> [(neighbor, weight), ...]

    def add_vertex(self, vertex):
        self.vertices.add(vertex)
        if vertex not in self.edges:
            self.edges[vertex] = []

    def add_edge(self, v1, v2, weight):
        self.add_vertex(v1)
        self.add_vertex(v2)
        self.edges[v1].append((v2, weight))
        self.edges[v2].append((v1, weight))  # Undirected graph

    def get_neighbors(self, vertex):
        return self.edges.get(vertex, [])


# Example usage
def main():
    # Create grid-based space
    grid_space = GridBasedSpace(width=20, height=20, resolution=0.5)

    # Add obstacles
    grid_space.add_obstacle(5, 0, 2, 4)
    grid_space.add_obstacle(-3, 3, 3, 2)
    grid_space.add_obstacle(0, -4, 4, 2)

    # Create graph representation
    graph = Graph()

    # Add all free cells as vertices
    for i in range(grid_space.grid.shape[1]):
        for j in range(grid_space.grid.shape[0]):
            if grid_space.is_valid_cell(i, j):
                graph.add_vertex((i, j))

    # Connect neighboring cells with edges
    for vertex in graph.vertices:
        x, y = vertex
        neighbors = grid_space.get_neighbors(x, y)
        for neighbor in neighbors:
            # Calculate distance as weight
            dist = np.sqrt((neighbor[0] - x)**2 + (neighbor[1] - y)**2)
            graph.add_edge(vertex, neighbor, dist)

    print(f"Graph created with {len(graph.vertices)} vertices and {sum(len(edges) for edges in graph.edges.values())} edges")


if __name__ == '__main__':
    main()
```

## Classical Path Planning Algorithms

### Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path from a start node to all other nodes in a weighted graph:

```python
import heapq
import numpy as np
import matplotlib.pyplot as plt


def dijkstra(graph, start, goal):
    """
    Dijkstra's algorithm for shortest path
    graph: Graph object with get_neighbors method
    start: Starting vertex
    goal: Goal vertex
    """
    # Priority queue: (cost, vertex)
    pq = [(0, start)]

    # Costs to reach each vertex
    costs = {start: 0}

    # Previous vertex in optimal path
    previous = {start: None}

    visited = set()

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        # If we reached the goal, reconstruct path
        if current == goal:
            path = []
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = previous[curr]
            return path[::-1], current_cost

        # Check neighbors
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in visited:
                continue

            new_cost = current_cost + weight

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                previous[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return None, float('inf')  # No path found


class GridGraph:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape

    def get_neighbors(self, pos):
        """Get valid neighbors with their weights (distances)"""
        x, y = pos
        neighbors = []

        # 8-connectivity
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Check bounds
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Check if cell is not an obstacle
                    if self.grid[ny, nx] == 0:  # 0 = free space
                        # Calculate distance (diagonal = sqrt(2), orthogonal = 1)
                        if dx != 0 and dy != 0:
                            weight = np.sqrt(2)
                        else:
                            weight = 1.0
                        neighbors.append(((nx, ny), weight))

        return neighbors


# Example usage
def main():
    # Create a grid with obstacles
    grid = np.zeros((10, 10))
    # Add some obstacles
    grid[3:6, 4:6] = 1  # Wall
    grid[1:3, 7:9] = 1  # Another obstacle

    graph = GridGraph(grid)

    start = (1, 1)
    goal = (8, 8)

    path, cost = dijkstra(graph, start, goal)

    if path:
        print(f"Dijkstra path found with cost: {cost:.2f}")
        print(f"Path: {path}")

        # Visualize the result
        plt.figure(figsize=(10, 10))

        # Plot grid
        plt.imshow(grid, cmap='binary', origin='upper')

        # Plot path
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, 'r-', linewidth=3, label='Dijkstra Path')
            plt.plot(path_y[0], path_x[0], 'go', markersize=10, label='Start')
            plt.plot(path_y[-1], path_x[-1], 'ro', markersize=10, label='Goal')

        plt.title(f'Dijkstra Path Planning\nCost: {cost:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No path found!")


if __name__ == '__main__':
    main()
```

### A* Algorithm

A* is an extension of Dijkstra's algorithm that uses a heuristic to guide the search toward the goal:

```python
import heapq
import numpy as np
import matplotlib.pyplot as plt


def manhattan_distance(pos1, pos2):
    """Manhattan distance heuristic"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1, pos2):
    """Euclidean distance heuristic"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def astar(graph, start, goal, heuristic_func=euclidean_distance):
    """
    A* algorithm for shortest path
    graph: Graph object with get_neighbors method
    start: Starting vertex
    goal: Goal vertex
    heuristic_func: Heuristic function h(n) that estimates cost from n to goal
    """
    # Priority queue: (f_score, g_score, vertex)
    pq = [(heuristic_func(start, goal), 0, start)]

    # Costs to reach each vertex (g_scores)
    g_scores = {start: 0}

    # Previous vertex in optimal path
    previous = {start: None}

    visited = set()

    while pq:
        f_score, g_score, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        # If we reached the goal, reconstruct path
        if current == goal:
            path = []
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = previous[curr]
            return path[::-1], g_score

        # Check neighbors
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in visited:
                continue

            tentative_g_score = g_score + weight

            if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_func(neighbor, goal)
                previous[neighbor] = current
                heapq.heappush(pq, (f_score, tentative_g_score, neighbor))

    return None, float('inf')  # No path found


# Example usage with comparison to Dijkstra
def main():
    # Create a grid with obstacles
    grid = np.zeros((15, 15))
    # Add some obstacles
    grid[5:8, 6:8] = 1  # Vertical wall
    grid[2:4, 10:12] = 1  # Horizontal obstacle
    grid[10:12, 3:7] = 1  # Another wall

    graph = GridGraph(grid)

    start = (1, 1)
    goal = (13, 13)

    # Run A* with different heuristics
    path_euc, cost_euc = astar(graph, start, goal, euclidean_distance)
    path_man, cost_man = astar(graph, start, goal, manhattan_distance)

    print(f"A* (Euclidean) path cost: {cost_euc:.2f}")
    print(f"A* (Manhattan) path cost: {cost_man:.2f}")

    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # A* with Euclidean heuristic
    axes[0].imshow(grid, cmap='binary', origin='upper')
    if path_euc:
        path_x, path_y = zip(*path_euc)
        axes[0].plot(path_y, path_x, 'r-', linewidth=3, label='A* Path')
        axes[0].plot(path_y[0], path_x[0], 'go', markersize=10, label='Start')
        axes[0].plot(path_y[-1], path_x[-1], 'ro', markersize=10, label='Goal')
    axes[0].set_title(f'A* with Euclidean Heuristic\nCost: {cost_euc:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # A* with Manhattan heuristic
    axes[1].imshow(grid, cmap='binary', origin='upper')
    if path_man:
        path_x, path_y = zip(*path_man)
        axes[1].plot(path_y, path_x, 'b-', linewidth=3, label='A* Path')
        axes[1].plot(path_y[0], path_x[0], 'go', markersize=10, label='Start')
        axes[1].plot(path_y[-1], path_x[-1], 'ro', markersize=10, label='Goal')
    axes[1].set_title(f'A* with Manhattan Heuristic\nCost: {cost_man:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```

### Rapidly-exploring Random Trees (RRT)

RRT is a probabilistically complete motion planning algorithm particularly useful for high-dimensional configuration spaces:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


class RRT:
    def __init__(self, start, goal, bounds, obstacle_list, max_iter=1000, step_size=0.5):
        """
        Initialize RRT planner
        start: Start configuration
        goal: Goal configuration
        bounds: List of (min, max) for each dimension
        obstacle_list: List of obstacles (for collision checking)
        max_iter: Maximum number of iterations
        step_size: Step size for extending tree
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.obstacles = obstacle_list
        self.max_iter = max_iter
        self.step_size = step_size

        # Tree represented as dictionary: child -> parent
        self.tree = {tuple(self.start): None}

        # For visualization
        self.nodes = [self.start.copy()]

    def is_collision_free(self, point):
        """Check if point is collision-free"""
        x, y = point

        # Check bounds
        if not (self.bounds[0][0] <= x <= self.bounds[0][1] and
                self.bounds[1][0] <= y <= self.bounds[1][1]):
            return False

        # Check obstacles (simplified circular obstacles)
        for obs_center, obs_radius in self.obstacles:
            if euclidean(point, obs_center) < obs_radius:
                return False

        return True

    def random_sample(self):
        """Randomly sample a point in configuration space"""
        x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        return np.array([x, y])

    def nearest_node(self, point):
        """Find nearest node in tree to given point"""
        min_dist = float('inf')
        nearest = None

        for node in self.tree:
            dist = euclidean(point, node)
            if dist < min_dist:
                min_dist = dist
                nearest = np.array(node)

        return nearest

    def extend_toward(self, from_node, to_point):
        """Extend tree from from_node toward to_point by step_size"""
        direction = to_point - from_node
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            # If close enough, connect directly
            new_point = to_point
        else:
            # Otherwise, extend by step_size in the direction
            direction = direction / distance  # Normalize
            new_point = from_node + direction * self.step_size

        # Check if the new point is collision-free
        if self.is_collision_free(new_point):
            # Also check if the path between from_node and new_point is collision-free
            if self.path_is_collision_free(from_node, new_point):
                return new_point

        return None

    def path_is_collision_free(self, start, end, num_samples=10):
        """Check if path between start and end is collision-free"""
        for i in range(1, num_samples + 1):
            t = i / num_samples
            point = start + t * (end - start)
            if not self.is_collision_free(point):
                return False
        return True

    def connect_to_goal(self, node):
        """Try to connect a node directly to the goal"""
        if self.path_is_collision_free(node, self.goal):
            return True
        return False

    def plan(self):
        """Plan path using RRT"""
        for i in range(self.max_iter):
            # Randomly sample a point
            if np.random.random() < 0.05:  # 5% chance to sample goal
                rand_point = self.goal
            else:
                rand_point = self.random_sample()

            # Find nearest node in tree
            nearest = self.nearest_node(rand_point)

            # Try to extend toward the random point
            new_point = self.extend_toward(nearest, rand_point)

            if new_point is not None:
                # Add new node to tree
                new_tuple = tuple(new_point)
                self.tree[new_tuple] = tuple(nearest)
                self.nodes.append(new_point)

                # Try to connect to goal
                if self.connect_to_goal(new_point):
                    # Goal reached, reconstruct path
                    return self.reconstruct_path(tuple(new_point))

        # If max iterations reached without finding path
        return None

    def reconstruct_path(self, goal_node):
        """Reconstruct path from goal to start"""
        path = []
        current = goal_node

        while current is not None:
            path.append(np.array(current))
            current = self.tree[current]

        return path[::-1]  # Reverse to get start->goal path


# Example usage
def main():
    # Define environment
    bounds = [(-10, 10), (-10, 10)]  # x and y bounds
    obstacles = [
        ((0, 0), 2),      # Circular obstacle at (0,0) with radius 2
        ((5, 5), 1.5),    # Circular obstacle at (5,5) with radius 1.5
        ((-5, -3), 1),    # Circular obstacle at (-5,-3) with radius 1
    ]

    start = np.array([-8, -8])
    goal = np.array([8, 8])

    # Create RRT planner
    rrt = RRT(start, goal, bounds, obstacles, max_iter=2000, step_size=0.5)

    # Plan path
    path = rrt.plan()

    if path:
        print(f"RRT path found with {len(path)} waypoints")

        # Visualize results
        plt.figure(figsize=(12, 10))

        # Plot obstacles
        for obs_center, obs_radius in obstacles:
            circle = plt.Circle(obs_center, obs_radius, color='red', alpha=0.3)
            plt.gca().add_patch(circle)

        # Plot tree nodes
        nodes_array = np.array(rrt.nodes)
        plt.scatter(nodes_array[:, 0], nodes_array[:, 1], s=1, c='lightblue', alpha=0.5, label='Tree Nodes')

        # Plot tree edges
        for child, parent in rrt.tree.items():
            if parent is not None:
                plt.plot([parent[0], child[0]], [parent[1], child[1]], 'lightblue', alpha=0.3, linewidth=0.5)

        # Plot path
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, label='RRT Path')
            plt.scatter(path_array[0, 0], path_array[0, 1], c='green', s=100, zorder=5, label='Start')
            plt.scatter(path_array[-1, 0], path_array[-1, 1], c='red', s=100, zorder=5, label='Goal')

        plt.xlim(bounds[0][0]-1, bounds[0][1]+1)
        plt.ylim(bounds[1][0]-1, bounds[1][1]+1)
        plt.title('RRT Path Planning')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    else:
        print("No path found with RRT!")


if __name__ == '__main__':
    main()
```

## Sampling-Based Motion Planning

### Probabilistic Roadmaps (PRM)

Probabilistic Roadmaps precompute a roadmap of the free space and use it for path planning:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict


class PRM:
    def __init__(self, bounds, obstacles, num_samples=500, connection_radius=1.5):
        """
        Initialize Probabilistic Roadmap
        bounds: List of (min, max) for each dimension
        obstacles: List of obstacles for collision checking
        num_samples: Number of random samples to generate
        connection_radius: Maximum distance to connect nodes
        """
        self.bounds = bounds
        self.obstacles = obstacles
        self.num_samples = num_samples
        self.connection_radius = connection_radius

        # Graph representation
        self.graph = defaultdict(list)  # node -> [neighbors]
        self.nodes = []  # List of all nodes

        # Build the roadmap
        self.build_roadmap()

    def is_collision_free(self, point):
        """Check if point is collision-free"""
        x, y = point

        # Check bounds
        if not (self.bounds[0][0] <= x <= self.bounds[0][1] and
                self.bounds[1][0] <= y <= self.bounds[1][1]):
            return False

        # Check obstacles
        for obs_center, obs_radius in self.obstacles:
            if np.linalg.norm(point - obs_center) < obs_radius:
                return False

        return True

    def build_roadmap(self):
        """Build the probabilistic roadmap"""
        # Sample random configurations
        valid_configs = []

        while len(valid_configs) < self.num_samples:
            # Sample random point
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
            point = np.array([x, y])

            if self.is_collision_free(point):
                valid_configs.append(point)

        self.nodes = valid_configs

        # Create KD-tree for efficient neighbor lookup
        if self.nodes:
            nodes_array = np.array(self.nodes)
            self.kdtree = cKDTree(nodes_array)

            # Connect nearby nodes
            for i, node in enumerate(self.nodes):
                # Find all nodes within connection radius
                indices = self.kdtree.query_ball_point(node, self.connection_radius)

                for j in indices:
                    if i != j:  # Don't connect to self
                        neighbor = self.nodes[j]

                        # Check if path between nodes is collision-free
                        if self.path_is_collision_free(node, neighbor):
                            # Add edge to graph (bidirectional)
                            self.graph[i].append(j)
                            self.graph[j].append(i)

    def path_is_collision_free(self, start, end, num_samples=10):
        """Check if path between start and end is collision-free"""
        for i in range(1, num_samples + 1):
            t = i / num_samples
            point = start + t * (end - start)
            if not self.is_collision_free(point):
                return False
        return True

    def find_path(self, start, goal):
        """
        Find path between start and goal using the roadmap
        start, goal: Start and goal configurations
        """
        # Find nearest nodes in roadmap to start and goal
        start_idx = self.find_nearest_node(start)
        goal_idx = self.find_nearest_node(goal)

        # Add start and goal temporarily to graph
        temp_graph = self.graph.copy()
        temp_nodes = self.nodes[:]

        # Connect start to nearby roadmap nodes
        start_connections = []
        start_neighbors = self.kdtree.query_ball_point(start, self.connection_radius)
        for neighbor_idx in start_neighbors:
            neighbor = self.nodes[neighbor_idx]
            if self.path_is_collision_free(start, neighbor):
                start_connections.append(neighbor_idx)

        # Connect goal to nearby roadmap nodes
        goal_connections = []
        goal_neighbors = self.kdtree.query_ball_point(goal, self.connection_radius)
        for neighbor_idx in goal_neighbors:
            neighbor = self.nodes[neighbor_idx]
            if self.path_is_collision_free(goal, neighbor):
                goal_connections.append(neighbor_idx)

        # Use A* to find path through the roadmap
        path_indices = self.astar_search(temp_graph, start_idx, goal_idx, start_connections, goal_connections)

        if path_indices:
            # Convert indices back to coordinates
            path = [start]  # Start with actual start
            for idx in path_indices[1:-1]:  # Skip first (duplicate) and last (will add goal separately)
                path.append(temp_nodes[idx])
            path.append(goal)  # End with actual goal
            return path
        else:
            return None

    def find_nearest_node(self, point):
        """Find index of nearest node to given point"""
        if not self.nodes:
            return -1

        distances, indices = self.kdtree.query(point, k=1)
        return indices

    def astar_search(self, graph, start_idx, goal_idx, start_connections, goal_connections):
        """A* search on the graph"""
        import heapq

        # Priority queue: (f_score, g_score, node_idx)
        pq = [(0, 0, start_idx)]

        g_scores = {start_idx: 0}
        previous = {start_idx: None}
        visited = set()

        while pq:
            f_score, g_score, current_idx = heapq.heappop(pq)

            if current_idx in visited:
                continue

            visited.add(current_idx)

            # Check if we reached the goal
            if current_idx == goal_idx:
                # Reconstruct path
                path = []
                curr_idx = goal_idx
                while curr_idx is not None:
                    path.append(curr_idx)
                    curr_idx = previous[curr_idx]
                return path[::-1]

            # Get neighbors
            neighbors = graph[current_idx][:]

            # If current node is the start, add connections to start_connections
            if current_idx == start_idx:
                neighbors.extend(start_connections)
            # If current node is in goal_connections, add goal as neighbor
            elif current_idx in goal_connections:
                neighbors.append(goal_idx)

            for neighbor_idx in neighbors:
                if neighbor_idx in visited:
                    continue

                # Calculate distance
                if neighbor_idx == goal_idx:
                    # Distance to actual goal
                    dist = np.linalg.norm(self.nodes[current_idx] - goal)
                elif current_idx == start_idx and neighbor_idx in start_connections:
                    # Distance from start to roadmap
                    dist = np.linalg.norm(start - self.nodes[neighbor_idx])
                else:
                    # Distance between roadmap nodes
                    dist = np.linalg.norm(self.nodes[current_idx] - self.nodes[neighbor_idx])

                tentative_g_score = g_score + dist

                if neighbor_idx not in g_scores or tentative_g_score < g_scores[neighbor_idx]:
                    g_scores[neighbor_idx] = tentative_g_score
                    # Heuristic: straight-line distance to goal
                    if neighbor_idx == goal_idx:
                        heuristic = 0
                    else:
                        heuristic = np.linalg.norm(self.nodes[neighbor_idx] - goal)

                    f_score = tentative_g_score + heuristic
                    previous[neighbor_idx] = current_idx
                    heapq.heappush(pq, (f_score, tentative_g_score, neighbor_idx))

        return None  # No path found


# Example usage
def main():
    # Define environment
    bounds = [(-10, 10), (-10, 10)]
    obstacles = [
        ((0, 0), 2),      # Circular obstacle at (0,0) with radius 2
        ((5, 5), 1.5),    # Circular obstacle at (5,5) with radius 1.5
        ((-5, -3), 1),    # Circular obstacle at (-5,-3) with radius 1
        ((3, -4), 1.2),   # Another obstacle
    ]

    start = np.array([-8, -8])
    goal = np.array([8, 8])

    # Create PRM
    prm = PRM(bounds, obstacles, num_samples=300, connection_radius=2.0)

    # Find path
    path = prm.find_path(start, goal)

    if path:
        print(f"PRM path found with {len(path)} waypoints")

        # Visualize results
        plt.figure(figsize=(12, 10))

        # Plot obstacles
        for obs_center, obs_radius in obstacles:
            circle = plt.Circle(obs_center, obs_radius, color='red', alpha=0.3)
            plt.gca().add_patch(circle)

        # Plot roadmap nodes
        nodes_array = np.array(prm.nodes)
        plt.scatter(nodes_array[:, 0], nodes_array[:, 1], s=1, c='lightblue', alpha=0.5, label='Roadmap Nodes')

        # Plot roadmap edges (sample some for visualization)
        plotted_edges = 0
        max_edges = 500  # Limit for visualization
        for node_idx, neighbors in prm.graph.items():
            for neighbor_idx in neighbors:
                if plotted_edges >= max_edges:
                    break
                plt.plot([prm.nodes[node_idx][0], prm.nodes[neighbor_idx][0]],
                        [prm.nodes[node_idx][1], prm.nodes[neighbor_idx][1]],
                        'lightblue', alpha=0.2, linewidth=0.5)
                plotted_edges += 1
            if plotted_edges >= max_edges:
                break

        # Plot path
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, label='PRM Path')
            plt.scatter(path_array[0, 0], path_array[0, 1], c='green', s=100, zorder=5, label='Start')
            plt.scatter(path_array[-1, 0], path_array[-1, 1], c='red', s=100, zorder=5, label='Goal')

        plt.xlim(bounds[0][0]-1, bounds[0][1]+1)
        plt.ylim(bounds[1][0]-1, bounds[1][1]+1)
        plt.title('Probabilistic Roadmap (PRM) Path Planning')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    else:
        print("No path found with PRM!")


if __name__ == '__main__':
    main()
```

## Navigation and Path Following

### Pure Pursuit Path Following

Pure pursuit is a classic path following algorithm that works well for car-like robots:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class PurePursuitController:
    def __init__(self, lookahead_distance=1.0):
        """
        Initialize pure pursuit controller
        lookahead_distance: Distance ahead to look for target point
        """
        self.lookahead_distance = lookahead_distance
        self.path = None
        self.current_path_index = 0

    def set_path(self, path):
        """Set the path to follow"""
        self.path = np.array(path)
        self.current_path_index = 0

    def find_lookahead_point(self, robot_pos):
        """Find the point on the path that is lookahead_distance ahead"""
        if self.path is None or len(self.path) == 0:
            return None

        # Start searching from current path index
        for i in range(self.current_path_index, len(self.path)):
            path_point = self.path[i]
            distance = np.linalg.norm(robot_pos - path_point)

            if distance >= self.lookahead_distance:
                # Update current path index for efficiency
                self.current_path_index = max(0, i - 1)
                return path_point

        # If no point is far enough, return the last point
        self.current_path_index = len(self.path) - 1
        return self.path[-1]

    def calculate_steering_angle(self, robot_pos, robot_heading, lookahead_point):
        """
        Calculate steering angle for pure pursuit
        robot_pos: Current robot position [x, y]
        robot_heading: Current robot heading (radians)
        lookahead_point: Target point to pursue
        """
        if lookahead_point is None:
            return 0.0

        # Vector from robot to lookahead point
        dx = lookahead_point[0] - robot_pos[0]
        dy = lookahead_point[1] - robot_pos[1]

        # Transform to robot's frame
        local_x = dx * np.cos(robot_heading) + dy * np.sin(robot_heading)
        local_y = -dx * np.sin(robot_heading) + dy * np.cos(robot_heading)

        # Calculate curvature (steering angle)
        # kappa = 2 * local_y / lookahead_distance^2
        if self.lookahead_distance > 0:
            curvature = 2 * local_y / (self.lookahead_distance ** 2)
            steering_angle = np.arctan(curvature * self.lookahead_distance)
        else:
            steering_angle = 0.0

        return steering_angle

    def follow_path(self, robot_pos, robot_heading, target_speed=1.0):
        """
        Follow the path and return control commands
        Returns: (steering_angle, linear_velocity)
        """
        lookahead_point = self.find_lookahead_point(robot_pos)
        steering_angle = self.calculate_steering_angle(robot_pos, robot_heading, lookahead_point)

        # Adjust speed based on curvature
        curvature = abs(steering_angle) / self.lookahead_distance if self.lookahead_distance > 0 else 0
        adjusted_speed = target_speed / (1 + 2 * curvature)  # Reduce speed in sharp turns

        return steering_angle, adjusted_speed


# Example simulation
def simulate_path_following():
    """Simulate a robot following a path using pure pursuit"""
    # Create a curved path
    t = np.linspace(0, 4*np.pi, 100)
    path_x = t * 0.5
    path_y = 2 * np.sin(t * 0.5)
    path = np.column_stack([path_x, path_y])

    # Initialize controller
    controller = PurePursuitController(lookahead_distance=1.5)
    controller.set_path(path)

    # Robot initial state
    robot_pos = np.array([0.0, 0.0])
    robot_heading = 0.0  # Robot initially facing along x-axis
    robot_speed = 0.0

    # Simulation parameters
    dt = 0.1
    simulation_time = 30.0
    t_sim = np.arange(0, simulation_time, dt)

    # Storage for robot trajectory
    robot_trajectory = [robot_pos.copy()]

    for t in t_sim:
        # Get control commands
        steering_angle, target_speed = controller.follow_path(robot_pos, robot_heading, target_speed=1.0)

        # Simple bicycle model for robot motion
        # Update robot state
        robot_speed = target_speed  # In this simple model, actual speed = target speed

        # Update position and heading
        robot_heading += (robot_speed * np.tan(steering_angle) / 2.0) * dt  # Simplified kinematic model
        robot_pos[0] += robot_speed * np.cos(robot_heading) * dt
        robot_pos[1] += robot_speed * np.sin(robot_heading) * dt

        # Store robot position
        robot_trajectory.append(robot_pos.copy())

    # Convert to array
    robot_trajectory = np.array(robot_trajectory)

    # Visualization
    plt.figure(figsize=(12, 8))

    # Plot path and robot trajectory
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Desired Path')
    plt.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], 'r-', linewidth=2, label='Robot Trajectory')
    plt.scatter(robot_trajectory[::10, 0], robot_trajectory[::10, 1], c='red', s=20, label='Robot Position', zorder=5)

    # Add start and end markers
    plt.scatter(path[0, 0], path[0, 1], c='green', s=100, marker='o', label='Path Start', zorder=5)
    plt.scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='s', label='Path End', zorder=5)
    plt.scatter(robot_trajectory[0, 0], robot_trajectory[0, 1], c='purple', s=100, marker='^', label='Robot Start', zorder=5)

    plt.title('Pure Pursuit Path Following')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    print(f"Path following completed. Robot traveled {len(robot_trajectory)} steps.")
    print(f"Final robot position: ({robot_trajectory[-1, 0]:.2f}, {robot_trajectory[-1, 1]:.2f})")
    print(f"Final path point: ({path[-1, 0]:.2f}, {path[-1, 1]:.2f})")
    print(f"Final distance from path end: {np.linalg.norm(robot_trajectory[-1] - path[-1]):.2f}")


if __name__ == '__main__':
    simulate_path_following()
```

### Dynamic Window Approach (DWA)

DWA is a local path planning algorithm that considers robot dynamics:

```python
import numpy as np
import matplotlib.pyplot as plt


class DynamicWindowApproach:
    def __init__(self, max_speed=1.0, min_speed=0.0, max_yawrate=40.0*np.pi/180.0,
                 max_accel=0.5, max_dyawrate=40.0*np.pi/180.0,
                 v_resolution=0.05, yawrate_resolution=0.1*np.pi/180.0,
                 dt=0.1, predict_time=3.0, to_goal_cost_gain=0.15, speed_cost_gain=1.0):
        """
        Initialize Dynamic Window Approach planner
        """
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yawrate = max_yawrate
        self.max_accel = max_accel
        self.max_dyawrate = max_dyawrate
        self.v_resolution = v_resolution
        self.yawrate_resolution = yawrate_resolution
        self.dt = dt
        self.predict_time = predict_time
        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_cost_gain = speed_cost_gain

    def motion(self, x, u, dt):
        """
        Motion model: update robot state
        x: [x, y, yaw, v, omega]
        u: [v, omega]
        """
        x[0] += u[0] * np.cos(x[2]) * dt
        x[1] += u[0] * np.sin(x[2]) * dt
        x[2] += u[1] * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_dynamic_window(self, x):
        """
        Calculate dynamic window
        x: current state [x, y, yaw, v, omega]
        """
        # Dynamic window from robot specification
        vs = [self.min_speed, self.max_speed,
              -self.max_yawrate, self.max_yawrate]

        # Dynamic window from motion model
        vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_dyawrate * self.dt,
              x[4] + self.max_dyawrate * self.dt]

        # Minimum window
        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]
        return dw

    def calc_trajectory(self, x_init, v, omega):
        """
        Calculate trajectory with given velocity and angular velocity
        """
        x = x_init.copy()
        trajectory = np.array(x)

        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, omega], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory

    def calc_to_goal_cost(self, trajectory, goal):
        """
        Calculate cost to goal
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = np.arctan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(np.arctan2(np.sin(cost_angle), np.cos(cost_angle)))

        return self.to_goal_cost_gain * cost

    def calc_obstacle_cost(self, trajectory, ob):
        """
        Calculate cost of obstacle proximity
        """
        min_r = float("inf")
        for ii in range(len(trajectory)):
            for i in range(len(ob)):
                ox = ob[i, 0]
                oy = ob[i, 1]
                dx = trajectory[ii, 0] - ox
                dy = trajectory[ii, 1] - oy
                r = np.sqrt(dx**2 + dy**2)
                if r <= min_r:
                    min_r = r

        return 1.0 / min_r if min_r != 0 else float("inf")

    def calc_control_and_trajectory(self, x, dw, goal, ob):
        """
        Calculate the best control command and trajectory
        """
        x_init = x.copy()
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # Evaluate all trajectory candidates
        v = dw[0]
        while v <= dw[1]:
            omega = dw[2]
            while omega <= dw[3]:
                # Calculate candidate trajectory
                trajectory = self.calc_trajectory(x_init, v, omega)

                # Calculate all costs
                to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                ob_cost = self.calc_obstacle_cost(trajectory, ob)

                # Total cost
                final_cost = to_goal_cost + speed_cost + ob_cost

                # Search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, omega]
                    best_trajectory = trajectory

                omega += self.yawrate_resolution
            v += self.v_resolution

        return best_u, best_trajectory

    def plan(self, x, goal, ob):
        """
        Plan path using Dynamic Window Approach
        x: initial state [x, y, yaw, v, omega]
        goal: goal position [x, y]
        ob: obstacle positions [[x1, y1], [x2, y2], ...]
        """
        # Calculate dynamic window
        dw = self.calc_dynamic_window(x)

        # Calculate control command and trajectory
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, ob)

        return u, trajectory


# Example usage
def main():
    # Initialize DWA
    dwa = DynamicWindowApproach()

    # Initial state [x, y, yaw, v, omega]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Goal position
    goal = np.array([10.0, 10.0])

    # Obstacles [[x1, y1], [x2, y2], ...]
    ob = np.array([
        [1, 2],
        [5, 3],
        [8, 4],
        [2, 7],
        [3, 8],
        [7, 9]
    ])

    # Simulation parameters
    dt = 0.1
    simulation_time = 50.0
    time = 0

    # Storage for trajectory
    trajectory = np.array(x)

    print("Starting DWA simulation...")
    while time <= simulation_time:
        # Plan using DWA
        u, predicted_trajectory = dwa.plan(x, goal, ob)

        # Update state
        x = dwa.motion(x, u, dt)

        # Store state
        trajectory = np.vstack((trajectory, x))

        # Check goal
        dist_to_goal = np.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2)
        if dist_to_goal <= 1.0:  # Goal reached
            print("Goal reached!")
            break

        time += dt

    # Visualization
    plt.figure(figsize=(12, 10))

    # Plot obstacles
    plt.plot(ob[:, 0], ob[:, 1], 'ko', markersize=10, label='Obstacles')

    # Plot goal
    plt.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal')

    # Plot trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Robot Trajectory')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')

    # Add arrow to show robot orientation
    for i in range(0, len(trajectory), 10):  # Every 10th point
        x_pos, y_pos, yaw = trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]
        dx = 0.5 * np.cos(yaw)
        dy = 0.5 * np.sin(yaw)
        plt.arrow(x_pos, y_pos, dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

    plt.title('Dynamic Window Approach (DWA) Path Planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    print(f"Simulation completed. Robot traveled {len(trajectory)} steps.")
    print(f"Final robot position: ({x[0]:.2f}, {x[1]:.2f})")
    print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Final distance from goal: {dist_to_goal:.2f}")


if __name__ == '__main__':
    main()
```

## ROS 2 Integration for Navigation

### Navigation Stack Components

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np


class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_goal = None
        self.current_position = None
        self.current_yaw = 0.0
        self.map_data = None
        self.scan_data = None
        self.global_path = []
        self.local_path = []

        # Navigation parameters
        self.linear_speed = 0.5
        self.angular_speed = 0.5
        self.arrival_threshold = 0.5  # meters
        self.rotation_threshold = 0.1  # radians

        # Timer for navigation control
        self.nav_timer = self.create_timer(0.1, self.navigation_control)

        self.get_logger().info('Navigation controller initialized')

    def goal_callback(self, msg):
        """Handle new goal"""
        self.current_goal = msg.pose
        self.get_logger().info(f'New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        # Plan global path
        if self.map_data and self.current_position:
            self.plan_global_path()

    def odom_callback(self, msg):
        """Update current position from odometry"""
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Extract yaw from quaternion
        orientation = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(orientation)

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.scan_data = msg

    def map_callback(self, msg):
        """Update map data"""
        self.map_data = msg
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height} at {msg.info.resolution:.2f}m/cell')

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def plan_global_path(self):
        """Plan global path using the map"""
        if not self.current_position or not self.current_goal:
            return

        start = self.current_position
        goal = (self.current_goal.position.x, self.current_goal.position.y)

        # For demonstration, we'll create a simple path
        # In practice, you'd use A*, Dijkstra, or other path planning algorithms
        path = self.create_straight_line_path(start, goal)

        # Publish global plan
        self.publish_path(path, self.global_plan_pub, 'map', 'global_plan')
        self.global_path = path

    def create_straight_line_path(self, start, goal):
        """Create a straight-line path for demonstration"""
        path = []
        steps = max(int(np.linalg.norm(np.array(goal) - np.array(start)) / 0.5), 10)

        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append((x, y))

        return path

    def publish_path(self, path, publisher, frame_id, path_name):
        """Publish path to ROS"""
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        publisher.publish(path_msg)

    def navigation_control(self):
        """Main navigation control loop"""
        if not self.current_position or not self.current_goal:
            return

        # Check if we have a path to follow
        if self.global_path:
            # Get next waypoint from global path
            target_point = self.get_next_waypoint()

            if target_point:
                # Calculate control commands
                cmd_vel = self.calculate_control(target_point)

                # Publish command
                self.cmd_vel_pub.publish(cmd_vel)

                # Check if we've reached the goal
                dist_to_goal = np.linalg.norm(
                    np.array(self.current_position) -
                    np.array([self.current_goal.position.x, self.current_goal.position.y])
                )

                if dist_to_goal < self.arrival_threshold:
                    self.get_logger().info('Goal reached!')
                    self.stop_robot()
                    self.current_goal = None
                    self.global_path = []

    def get_next_waypoint(self):
        """Get the next waypoint to follow"""
        if not self.global_path:
            return None

        # For simplicity, follow the path in order
        # In practice, you'd use a more sophisticated approach like pure pursuit
        if len(self.global_path) > 0:
            return self.global_path[0]  # First point in path
        return None

    def calculate_control(self, target_point):
        """Calculate control commands to reach target point"""
        cmd_vel = Twist()

        if not self.current_position:
            return cmd_vel

        # Calculate vector to target
        dx = target_point[0] - self.current_position[0]
        dy = target_point[1] - self.current_position[1]

        # Calculate distance and angle to target
        distance = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)

        # Calculate angle difference
        angle_diff = target_angle - self.current_yaw
        # Normalize angle to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Simple proportional controller
        if distance > self.arrival_threshold:
            cmd_vel.linear.x = min(self.linear_speed, distance * 0.5)  # Scale speed with distance
            cmd_vel.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2.0))
        else:
            # Reached waypoint, move to next
            if len(self.global_path) > 0:
                self.global_path.pop(0)  # Remove current waypoint

        return cmd_vel

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    nav_controller = NavigationController()

    try:
        rclpy.spin(nav_controller)
    except KeyboardInterrupt:
        pass
    finally:
        nav_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: Autonomous Navigation System

### Objective
Create a complete autonomous navigation system that integrates global path planning, local path planning, and obstacle avoidance.

### Prerequisites
- Completed Chapter 1-9
- ROS 2 Humble with Navigation2 stack
- Basic understanding of motion planning and control

### Steps

1. **Create a navigation lab package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python navigation_lab --dependencies rclpy geometry_msgs nav_msgs sensor_msgs tf2_ros std_msgs numpy scipy matplotlib
   ```

2. **Create the main navigation node** (`navigation_lab/navigation_lab/navigation_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped, Twist
   from nav_msgs.msg import Path, OccupancyGrid
   from sensor_msgs.msg import LaserScan
   from visualization_msgs.msg import Marker, MarkerArray
   from std_msgs.msg import Bool
   import numpy as np
   import math


   class NavigationLabNode(Node):
       def __init__(self):
           super().__init__('navigation_lab_node')

           # Publishers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)
           self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
           self.obstacle_markers_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
           self.status_pub = self.create_publisher(Bool, '/navigation_active', 10)

           # Subscribers
           self.goal_sub = self.create_subscription(
               PoseStamped,
               '/goal_pose',
               self.goal_callback,
               10
           )

           self.odom_sub = self.create_subscription(
               Odometry,
               '/odom',
               self.odom_callback,
               10
           )

           self.scan_sub = self.create_subscription(
               LaserScan,
               '/scan',
               self.scan_callback,
               10
           )

           # Navigation state
           self.current_goal = None
           self.current_position = np.array([0.0, 0.0])
           self.current_yaw = 0.0
           self.scan_data = None
           self.navigation_active = False

           # Navigation parameters
           self.linear_speed = 0.5
           self.angular_speed = 0.5
           self.arrival_threshold = 0.5
           self.safe_distance = 0.8
           self.lookahead_distance = 1.0

           # Path planning
           self.global_path = []
           self.local_path = []
           self.path_index = 0

           # Controllers
           self.pure_pursuit = PurePursuitController(lookahead_distance=self.lookahead_distance)
           self.dwa_planner = DynamicWindowApproach()

           # Timer for navigation control
           self.nav_timer = self.create_timer(0.1, self.navigation_control)

           self.get_logger().info('Navigation lab node initialized')

       def goal_callback(self, msg):
           """Handle new goal"""
           goal_pos = np.array([msg.pose.position.x, msg.pose.position.y])
           self.current_goal = goal_pos
           self.get_logger().info(f'New goal received: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})')

           # Plan global path
           if self.current_position is not None:
               self.plan_global_path()

       def odom_callback(self, msg):
           """Update current position from odometry"""
           self.current_position[0] = msg.pose.pose.position.x
           self.current_position[1] = msg.pose.pose.position.y

           # Extract yaw from quaternion
           orientation = msg.pose.pose.orientation
           siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
           cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
           self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

       def scan_callback(self, msg):
           """Update laser scan data"""
           self.scan_data = msg

       def plan_global_path(self):
           """Plan global path to goal"""
           if self.current_goal is None:
               return

           # For this lab, create a simple path (in practice, use A*, RRT, etc.)
           start = self.current_position.copy()
           goal = self.current_goal.copy()

           # Create path points
           steps = max(int(np.linalg.norm(goal - start) / 0.5), 10)
           self.global_path = []

           for i in range(steps + 1):
               t = i / steps
               point = start + t * (goal - start)
               self.global_path.append(point)

           self.path_index = 0
           self.navigation_active = True

           # Publish global path
           self.publish_path(self.global_path, self.global_plan_pub, 'odom', 'global_plan')
           self.get_logger().info(f'Global path planned with {len(self.global_path)} waypoints')

       def navigation_control(self):
           """Main navigation control loop"""
           if not self.navigation_active or self.current_goal is None:
               self.publish_status(False)
               return

           # Check if goal is reached
           dist_to_goal = np.linalg.norm(self.current_position - self.current_goal)
           if dist_to_goal < self.arrival_threshold:
               self.get_logger().info('Goal reached!')
               self.stop_robot()
               self.navigation_active = False
               self.publish_status(False)
               return

           # Get local path based on global path and obstacles
           local_path = self.generate_local_path()
           self.local_path = local_path

           # Publish local path
           if local_path:
               self.publish_path(local_path, self.local_plan_pub, 'odom', 'local_plan')

           # Calculate control commands
           cmd_vel = self.calculate_control(local_path)
           self.cmd_vel_pub.publish(cmd_vel)

           # Publish navigation status
           self.publish_status(True)

       def generate_local_path(self):
           """Generate local path considering obstacles"""
           if not self.global_path or self.path_index >= len(self.global_path):
               return []

           # In a real system, this would use local planners like DWA or TEB
           # For this lab, we'll create a simple local path from current global path
           local_path = []
           start_idx = self.path_index

           # Take next few points from global path
           for i in range(start_idx, min(start_idx + 10, len(self.global_path))):
               local_path.append(self.global_path[i])

           # If we have scan data, consider obstacle avoidance
           if self.scan_data:
               # Convert laser scan to obstacle points
               obstacles = self.scan_to_obstacles()

               # Simple obstacle avoidance: if obstacle is close, adjust path
               if self.has_close_obstacle():
                   # Create detour path (simplified)
                   current_pos = self.current_position
                   goal_pos = self.global_path[min(self.path_index + 5, len(self.global_path)-1)] if self.global_path else self.current_goal

                   # Calculate detour point to the side
                   to_goal = goal_pos - current_pos
                   perpendicular = np.array([-to_goal[1], to_goal[0]])  # Perpendicular vector
                   perpendicular = perpendicular / np.linalg.norm(perpendicular) * 1.0  # Normalize and scale

                   detour_point = current_pos + perpendicular
                   local_path = [current_pos, detour_point, goal_pos]

           return local_path

       def scan_to_obstacles(self):
           """Convert laser scan to obstacle points"""
           if not self.scan_data:
               return []

           obstacles = []
           angle = self.scan_data.angle_min

           for range_val in self.scan_data.ranges:
               if not math.isnan(range_val) and range_val < 3.0:  # Valid range and within 3m
                   x = range_val * math.cos(angle)
                   y = range_val * math.sin(angle)
                   obstacles.append(np.array([x, y]))
               angle += self.scan_data.angle_increment

           return obstacles

       def has_close_obstacle(self):
           """Check if there are close obstacles"""
           if not self.scan_data:
               return False

           # Check for obstacles within safe distance
           close_obstacles = [r for r in self.scan_data.ranges
                             if not math.isnan(r) and r < self.safe_distance]
           return len(close_obstacles) > 0

       def calculate_control(self, local_path):
           """Calculate control commands to follow path"""
           cmd_vel = Twist()

           if not local_path:
               return cmd_vel

           # Use pure pursuit for path following
           if len(local_path) > 0:
               # Set the path for pure pursuit controller
               self.pure_pursuit.set_path(local_path)

               # Get control command
               robot_pos = self.current_position
               robot_heading = self.current_yaw
               steering_angle, linear_vel = self.pure_pursuit.follow_path(
                   robot_pos, robot_heading, target_speed=self.linear_speed
               )

               cmd_vel.linear.x = linear_vel
               cmd_vel.angular.z = steering_angle

           # Also consider obstacle avoidance
           if self.has_close_obstacle():
               # Emergency stop or evasive maneuver
               cmd_vel.linear.x *= 0.5  # Slow down
               # Add slight turning to avoid obstacles
               cmd_vel.angular.z += np.random.uniform(-0.2, 0.2)  # Random small turn

           return cmd_vel

       def publish_path(self, path_points, publisher, frame_id, path_name):
           """Publish path as Path message"""
           path_msg = Path()
           path_msg.header.frame_id = frame_id
           path_msg.header.stamp = self.get_clock().now().to_msg()

           for point in path_points:
               pose = PoseStamped()
               pose.pose.position.x = point[0]
               pose.pose.position.y = point[1]
               pose.pose.position.z = 0.0
               path_msg.poses.append(pose)

           publisher.publish(path_msg)

       def publish_status(self, active):
           """Publish navigation status"""
           status_msg = Bool()
           status_msg.data = active
           self.status_pub.publish(status_msg)

       def stop_robot(self):
           """Stop the robot"""
           cmd_vel = Twist()
           self.cmd_vel_pub.publish(cmd_vel)


   def main(args=None):
       rclpy.init(args=args)
       navigation_lab = NavigationLabNode()

       try:
           rclpy.spin(navigation_lab)
       except KeyboardInterrupt:
           pass
       finally:
           navigation_lab.stop_robot()
           navigation_lab.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`navigation_lab/launch/navigation_lab.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true'
       )

       # Navigation lab node
       navigation_lab_node = Node(
           package='navigation_lab',
           executable='navigation_node',
           name='navigation_lab_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           navigation_lab_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'navigation_lab'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Navigation lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'navigation_node = navigation_lab.navigation_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select navigation_lab
   source install/setup.bash
   ```

6. **Run the navigation system**:
   ```bash
   ros2 launch navigation_lab navigation_lab.launch.py
   ```

### Expected Results
- The robot should navigate from start to goal while avoiding obstacles
- Global and local paths should be published and visualized
- The system should handle dynamic obstacles appropriately
- Control commands should be published to drive the robot

### Troubleshooting Tips
- Ensure proper TF frames are available for navigation
- Check that laser scan data is being received correctly
- Verify that odometry is accurate for path following
- Monitor the logs for path planning and execution status

## Summary

In this chapter, we've explored the fundamental concepts of motion planning and navigation, including classical algorithms like Dijkstra and A*, sampling-based methods like RRT and PRM, and local navigation approaches like Pure Pursuit and Dynamic Window Approach. We've implemented practical examples of each concept and created a complete navigation system.

The hands-on lab provided experience with integrating global path planning, local path planning, and obstacle avoidance into a complete autonomous navigation system. This foundation is essential for more advanced robotic applications involving complex environments and dynamic obstacles, which we'll explore in the upcoming chapters.