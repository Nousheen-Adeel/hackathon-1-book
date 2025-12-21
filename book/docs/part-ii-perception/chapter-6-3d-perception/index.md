---
title: Chapter 6 - 3D Perception and Scene Understanding
sidebar_position: 3
---

# Chapter 6: 3D Perception and Scene Understanding

## Learning Goals

- Process 3D point cloud data
- Understand spatial reasoning and mapping
- Learn scene segmentation and understanding
- Process and visualize 3D point clouds
- Create 3D maps of environments
- Implement scene segmentation

## Introduction to 3D Perception

3D perception is a critical capability for robots operating in real-world environments. Unlike 2D computer vision, 3D perception provides geometric information about the environment, enabling robots to understand spatial relationships, navigate safely, and interact with objects in three-dimensional space.

### Point Clouds

Point clouds are collections of 3D points that represent the surface geometry of objects and environments. They are typically acquired using:

- **LIDAR sensors**: Provide accurate 3D measurements using laser ranging
- **RGB-D cameras**: Provide both color and depth information
- **Stereo cameras**: Generate depth through triangulation
- **Structured light sensors**: Project patterns to measure depth

### ROS 2 3D Perception Ecosystem

ROS 2 provides several packages for 3D perception:

- **sensor_msgs/PointCloud2**: Standard message type for point cloud data
- **PCL (Point Cloud Library)**: Extensive library for point cloud processing
- **rviz**: Visualization tool for 3D data
- **octomap**: 3D occupancy mapping
- **moveit**: Motion planning with 3D collision checking

## Point Cloud Processing Fundamentals

### Point Cloud Data Structure

Point clouds in ROS 2 use the `sensor_msgs/PointCloud2` message format, which is a flexible binary format that can store various types of point data (XYZ, XYZRGB, XYZI, etc.).

```python
# Basic point cloud processing in ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import numpy as np


class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/points_raw',  # Common topic name for point cloud data
            self.pointcloud_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Point cloud processor initialized')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        try:
            # Convert PointCloud2 message to list of points
            points = list(point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            ))

            # Convert to numpy array for processing
            points_array = np.array(points)

            if len(points_array) > 0:
                # Calculate basic statistics
                mean_point = np.mean(points_array, axis=0)
                std_point = np.std(points_array, axis=0)

                self.get_logger().info(
                    f'Point cloud: {len(points_array)} points, '
                    f'Mean: ({mean_point[0]:.2f}, {mean_point[1]:.2f}, {mean_point[2]:.2f}), '
                    f'Std: ({std_point[0]:.2f}, {std_point[1]:.2f}, {std_point[2]:.2f})'
                )

                # Example processing: filter points by distance
                filtered_points = self.filter_by_distance(points_array, max_distance=5.0)
                self.get_logger().info(f'Filtered points: {len(filtered_points)}')

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def filter_by_distance(self, points, max_distance=5.0):
        """Filter points based on distance from origin"""
        distances = np.linalg.norm(points[:, :3], axis=1)
        filtered_indices = distances <= max_distance
        return points[filtered_indices]


def main(args=None):
    rclpy.init(args=args)
    point_cloud_processor = PointCloudProcessor()
    rclpy.spin(point_cloud_processor)
    point_cloud_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Point Cloud Filtering

Point clouds often contain noise and outliers that need to be filtered before processing:

```python
import numpy as np
from scipy.spatial import cKDTree


class PointCloudFilter:
    def __init__(self):
        pass

    def statistical_outlier_removal(self, points, k=20, std_dev_thresh=2.0):
        """Remove statistical outliers from point cloud"""
        if len(points) < k:
            return points

        # Build k-d tree for neighbor search
        tree = cKDTree(points[:, :3])  # Use only x, y, z coordinates

        # Calculate distances to k nearest neighbors
        distances, _ = tree.query(points[:, :3], k=k+1)  # Include self in query

        # Calculate mean distance for each point (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude first distance (to self)

        # Calculate global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)

        # Filter points based on distance threshold
        valid_indices = mean_distances < (global_mean + std_dev_thresh * global_std)
        return points[valid_indices]

    def voxel_grid_filter(self, points, voxel_size=0.1):
        """Downsample point cloud using voxel grid filter"""
        # Calculate voxel coordinates
        voxel_coords = np.floor(points[:, :3] / voxel_size).astype(int)

        # Create unique voxel keys
        unique_voxels, indices = np.unique(voxel_coords, axis=0, return_index=True)

        # Return one point per voxel
        return points[indices]

    def radius_outlier_removal(self, points, radius=0.1, min_neighbors=2):
        """Remove points with few neighbors within radius"""
        if len(points) == 0:
            return points

        tree = cKDTree(points[:, :3])

        # Count neighbors within radius for each point
        neighbor_counts = tree.query_ball_point(points[:, :3], radius, return_length=True)

        # Keep points with sufficient neighbors
        valid_indices = neighbor_counts >= min_neighbors
        return points[valid_indices]

    def passthrough_filter(self, points, axis='z', min_val=-1.0, max_val=1.0):
        """Filter points based on axis limits"""
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        valid_indices = (points[:, axis_idx] >= min_val) & (points[:, axis_idx] <= max_val)
        return points[valid_indices]


# Example usage
def main():
    # Create synthetic point cloud data
    np.random.seed(42)

    # Generate some points in a plane
    x = np.random.uniform(-5, 5, 1000)
    y = np.random.uniform(-5, 5, 1000)
    z = np.random.normal(0, 0.1, 1000)  # Plane around z=0
    points = np.column_stack([x, y, z])

    # Add some noise/outliers
    outliers = np.random.uniform(-10, 10, (100, 3))
    points = np.vstack([points, outliers])

    print(f'Original points: {len(points)}')

    # Create filter instance
    filter_obj = PointCloudFilter()

    # Apply filters
    filtered_points = filter_obj.statistical_outlier_removal(points)
    print(f'After statistical outlier removal: {len(filtered_points)}')

    filtered_points = filter_obj.voxel_grid_filter(filtered_points, voxel_size=0.2)
    print(f'After voxel grid filtering: {len(filtered_points)}')

    filtered_points = filter_obj.passthrough_filter(filtered_points, 'z', -2.0, 2.0)
    print(f'After passthrough filtering: {len(filtered_points)}')


if __name__ == '__main__':
    main()
```

## Point Cloud Segmentation

### Ground Plane Segmentation

Ground plane segmentation is crucial for mobile robotics applications:

```python
import numpy as np
from sklearn.linear_model import RANSACRegressor


class GroundPlaneSegmenter:
    def __init__(self, distance_threshold=0.1, max_iterations=1000):
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def segment_ground_plane(self, points):
        """Segment ground plane using RANSAC"""
        if len(points) < 10:
            return np.array([]), np.array([])  # Not enough points

        # Prepare data for RANSAC (x, y, z -> fit z = ax + by + c)
        X = points[:, [0, 1]]  # x, y coordinates
        y = points[:, 2]       # z coordinates

        # Create and fit RANSAC model
        ransac = RANSACRegressor(
            min_samples=3,
            residual_threshold=self.distance_threshold,
            max_trials=self.max_iterations
        )
        ransac.fit(X, y)

        # Predict z values for all points
        predicted_z = ransac.predict(X)

        # Calculate distances from points to plane
        distances = np.abs(y - predicted_z)

        # Classify points as ground or obstacle
        ground_mask = distances < self.distance_threshold
        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]

        # Extract plane parameters (ax + by + cz + d = 0)
        a, b = ransac.estimator_.coef_
        c = -1
        d = ransac.estimator_.intercept_

        plane_params = np.array([a, b, c, d])

        return ground_points, obstacle_points, plane_params

    def segment_ground_plane_manual(self, points, max_slope=0.1, max_height=0.2):
        """Simple ground segmentation based on height and slope"""
        # Project points to 2D grid and find ground level
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Simple approach: find minimum z in grid cells
        grid_size = 0.5
        x_grid = np.floor(x / grid_size).astype(int)
        y_grid = np.floor(y / grid_size).astype(int)

        # Group points by grid cell and find minimum z
        grid_dict = {}
        for i in range(len(points)):
            grid_key = (x_grid[i], y_grid[i])
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(z[i])

        # Calculate ground level for each cell
        ground_levels = {}
        for grid_key, z_values in grid_dict.items():
            if len(z_values) > 3:  # Need enough points
                ground_levels[grid_key] = np.percentile(z_values, 10)  # Use 10th percentile

        # Classify points based on ground level
        ground_points = []
        obstacle_points = []

        for i in range(len(points)):
            grid_key = (x_grid[i], y_grid[i])
            if grid_key in ground_levels:
                ground_z = ground_levels[grid_key]
                if z[i] < ground_z + max_height:
                    ground_points.append(points[i])
                else:
                    obstacle_points.append(points[i])
            else:
                obstacle_points.append(points[i])

        return np.array(ground_points), np.array(obstacle_points)


# Example usage
def main():
    # Create synthetic data with ground plane and obstacles
    np.random.seed(42)

    # Ground plane points
    x_ground = np.random.uniform(-10, 10, 2000)
    y_ground = np.random.uniform(-10, 10, 2000)
    z_ground = np.random.normal(0, 0.05, 2000)  # Ground at z=0 with noise
    ground_points = np.column_stack([x_ground, y_ground, z_ground])

    # Add some obstacles (boxes, poles, etc.)
    # Box
    x_box = np.random.uniform(-2, -1, 100)
    y_box = np.random.uniform(1, 2, 100)
    z_box = np.random.uniform(0.1, 0.5, 100)
    box_points = np.column_stack([x_box, y_box, z_box])

    # Pole
    x_pole = np.random.uniform(3, 3.1, 50)
    y_pole = np.random.uniform(-1, -0.9, 50)
    z_pole = np.random.uniform(0.1, 1.0, 50)
    pole_points = np.column_stack([x_pole, y_pole, z_pole])

    # Combine all points
    all_points = np.vstack([ground_points, box_points, pole_points])

    print(f'Total points: {len(all_points)}')

    # Segment ground plane
    segmenter = GroundPlaneSegmenter(distance_threshold=0.1)
    ground, obstacles, plane_params = segmenter.segment_ground_plane(all_points)

    print(f'Ground points: {len(ground)}, Obstacle points: {len(obstacles)}')
    print(f'Ground plane parameters: {plane_params}')


if __name__ == '__main__':
    main()
```

### Euclidean Clustering

Euclidean clustering groups nearby points into objects:

```python
import numpy as np
from sklearn.cluster import DBSCAN


class EuclideanClusterer:
    def __init__(self, eps=0.3, min_points=10):
        self.eps = eps  # Maximum distance between points in same cluster
        self.min_points = min_points  # Minimum points to form a cluster

    def cluster_points(self, points):
        """Cluster points using DBSCAN"""
        if len(points) < self.min_points:
            return np.array([])  # Not enough points to cluster

        # Use only x, y, z coordinates for clustering
        coordinates = points[:, :3]

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_points)
        labels = clustering.fit_predict(coordinates)

        # Create clusters dictionary
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        # Convert to list of point indices for each cluster
        cluster_indices = [clusters[label] for label in clusters if label != -1]  # Exclude noise points

        return cluster_indices

    def extract_cluster_properties(self, points, cluster_indices):
        """Extract properties for each cluster"""
        cluster_properties = []

        for cluster_idx_list in cluster_indices:
            cluster_points = points[cluster_idx_list]

            # Calculate cluster properties
            centroid = np.mean(cluster_points, axis=0)
            size = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
            num_points = len(cluster_points)

            properties = {
                'centroid': centroid,
                'size': size,
                'num_points': num_points,
                'points': cluster_points,
                'indices': cluster_idx_list
            }

            cluster_properties.append(properties)

        return cluster_properties


# Example usage
def main():
    # Create synthetic data with multiple objects
    np.random.seed(42)

    # Object 1: Box
    x1 = np.random.uniform(0, 1, 100)
    y1 = np.random.uniform(0, 1, 100)
    z1 = np.random.uniform(0, 0.5, 100)
    object1 = np.column_stack([x1, y1, z1])

    # Object 2: Cylinder
    theta = np.random.uniform(0, 2*np.pi, 80)
    radius = np.random.uniform(0, 0.3, 80)
    x2 = 3 + radius * np.cos(theta)
    y2 = 2 + radius * np.sin(theta)
    z2 = np.random.uniform(0, 0.8, 80)
    object2 = np.column_stack([x2, y2, z2])

    # Object 3: Another box
    x3 = np.random.uniform(-2, -1, 120)
    y3 = np.random.uniform(-1, 0, 120)
    z3 = np.random.uniform(0, 0.6, 120)
    object3 = np.column_stack([x3, y3, z3])

    # Combine all objects
    all_points = np.vstack([object1, object2, object3])

    print(f'Total points: {len(all_points)}')

    # Cluster the points
    clusterer = EuclideanClusterer(eps=0.3, min_points=20)
    clusters = clusterer.cluster_points(all_points)

    print(f'Found {len(clusters)} clusters')

    # Extract cluster properties
    properties = clusterer.extract_cluster_properties(all_points, clusters)

    for i, prop in enumerate(properties):
        print(f'Cluster {i}: {prop["num_points"]} points, centroid at ({prop["centroid"][0]:.2f}, {prop["centroid"][1]:.2f}, {prop["centroid"][2]:.2f})')


if __name__ == '__main__':
    main()
```

## 3D Mapping and Reconstruction

### Occupancy Grid Mapping

Occupancy grid mapping represents the environment as a discrete grid of occupied/free/unknown states:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


class OccupancyGridMapper:
    def __init__(self, resolution=0.1, width=100, height=100):
        self.resolution = resolution  # meters per cell
        self.width = width  # number of cells
        self.height = height
        self.grid = np.full((height, width), 0.5, dtype=np.float32)  # 0.5 = unknown
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2

    def update_with_laser_scan(self, ranges, angles, robot_x, robot_y, robot_yaw):
        """Update occupancy grid with laser scan data"""
        # Convert scan to world coordinates
        for i, (range_val, angle) in enumerate(zip(ranges, angles)):
            if not (np.isfinite(range_val) and range_val > 0):
                continue

            # Calculate world coordinates of obstacle
            world_x = robot_x + range_val * np.cos(robot_yaw + angle)
            world_y = robot_y + range_val * np.sin(robot_yaw + angle)

            # Calculate cell coordinates
            cell_x = int((world_x - self.origin_x) / self.resolution)
            cell_y = int((world_y - self.origin_y) / self.resolution)

            # Check bounds
            if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                # Update cell as occupied
                self.grid[cell_y, cell_x] = 0.9  # highly occupied

                # Update free space along the ray
                num_steps = int(range_val / self.resolution)
                for step in range(num_steps):
                    ray_x = robot_x + (step * self.resolution) * np.cos(robot_yaw + angle)
                    ray_y = robot_y + (step * self.resolution) * np.sin(robot_yaw + angle)

                    ray_cell_x = int((ray_x - self.origin_x) / self.resolution)
                    ray_cell_y = int((ray_y - self.origin_y) / self.resolution)

                    if 0 <= ray_cell_x < self.width and 0 <= ray_cell_y < self.height:
                        # Update as free space (but don't override occupied areas)
                        if self.grid[ray_cell_y, ray_cell_x] > 0.3:
                            self.grid[ray_cell_y, ray_cell_x] = 0.2  # free space

    def get_grid_coordinates(self, world_x, world_y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def get_world_coordinates(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        world_x = self.origin_x + grid_x * self.resolution
        world_y = self.origin_y + grid_y * self.resolution
        return world_x, world_y

    def inflate_obstacles(self, inflation_radius=0.3):
        """Inflate obstacles by a certain radius"""
        inflation_cells = int(inflation_radius / self.resolution)

        # Create binary mask of occupied cells
        occupied_mask = self.grid > 0.7

        # Dilate the mask
        for _ in range(inflation_cells):
            occupied_mask = binary_dilation(occupied_mask)

        # Update grid with inflated obstacles
        self.grid[occupied_mask] = 0.9

    def visualize(self):
        """Visualize the occupancy grid"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='gray', origin='lower',
                  extent=[self.origin_x, self.origin_x + self.width * self.resolution,
                         self.origin_y, self.origin_y + self.height * self.resolution])
        plt.colorbar(label='Occupancy Probability')
        plt.title('Occupancy Grid Map')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage
def main():
    mapper = OccupancyGridMapper(resolution=0.1, width=200, height=200)

    # Simulate robot moving in a square path and taking laser scans
    robot_path = []
    for i in range(10):
        # Robot position
        robot_x = 2 * np.cos(i * 0.5)
        robot_y = 2 * np.sin(i * 0.5)
        robot_yaw = i * 0.5  # Robot orientation

        # Simulate laser scan (270 degree scan with 1 degree resolution)
        angles = np.deg2rad(np.arange(-135, 136, 1))
        ranges = np.full_like(angles, 5.0)  # Default range

        # Add some obstacles
        for j, angle in enumerate(angles):
            # Simulate an obstacle at (3, 0) relative to robot
            obs_x = 3.0
            obs_y = 0.0
            dist_to_obs = np.sqrt((obs_x - robot_x)**2 + (obs_y - robot_y)**2)
            angle_to_obs = np.arctan2(obs_y - robot_y, obs_x - robot_x) - robot_yaw

            # If laser ray points toward obstacle and is within range
            if abs(angle - angle_to_obs) < 0.2 and dist_to_obs < 5.0:
                ranges[j] = dist_to_obs

        # Update map with this scan
        mapper.update_with_laser_scan(ranges, angles, robot_x, robot_y, robot_yaw)
        robot_path.append((robot_x, robot_y))

    # Inflate obstacles
    mapper.inflate_obstacles(inflation_radius=0.3)

    # Visualize the map
    mapper.visualize()

    print(f'Map size: {mapper.width * mapper.resolution}m x {mapper.height * mapper.resolution}m')
    print(f'Resolution: {mapper.resolution}m per cell')


if __name__ == '__main__':
    main()
```

### 3D Reconstruction

3D reconstruction from multiple views or depth sensors:

```python
import numpy as np
import open3d as o3d  # This would be used in practice
from scipy.spatial import cKDTree


class Simple3DReconstructor:
    def __init__(self):
        pass

    def integrate_depth_images(self, depth_images, poses, camera_intrinsics):
        """Simple integration of depth images into a 3D model"""
        # This is a simplified version - in practice, you'd use TSDF integration
        # or other advanced techniques

        # Convert depth images to point clouds and transform to global frame
        all_points = []

        for depth_img, pose in zip(depth_images, poses):
            # Convert depth image to point cloud
            points = self.depth_to_pointcloud(depth_img, camera_intrinsics)

            # Transform points to global frame
            transformed_points = self.transform_points(points, pose)

            all_points.extend(transformed_points)

        return np.array(all_points)

    def depth_to_pointcloud(self, depth_img, camera_intrinsics):
        """Convert depth image to point cloud"""
        height, width = depth_img.shape

        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Convert pixel coordinates to camera coordinates
        x = (u_coords - camera_intrinsics['cx']) * depth_img / camera_intrinsics['fx']
        y = (v_coords - camera_intrinsics['cy']) * depth_img / camera_intrinsics['fy']
        z = depth_img

        # Stack into points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Remove invalid points (where depth is 0 or invalid)
        valid_mask = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
        return points[valid_mask]

    def transform_points(self, points, pose):
        """Transform points using 4x4 pose matrix"""
        # Add homogeneous coordinate
        points_h = np.hstack([points, np.ones((len(points), 1))])

        # Apply transformation
        transformed_h = (pose @ points_h.T).T

        # Remove homogeneous coordinate
        return transformed_h[:, :3]

    def voxel_grid_downsample(self, points, voxel_size=0.01):
        """Downsample point cloud using voxel grid"""
        # Calculate voxel coordinates
        voxel_coords = np.floor(points / voxel_size).astype(int)

        # Create unique voxel keys
        unique_voxels, indices = np.unique(voxel_coords, axis=0, return_index=True)

        # Return one point per voxel
        return points[indices]

    def estimate_normals(self, points, k=20):
        """Estimate surface normals for point cloud"""
        if len(points) < k:
            return np.array([])

        # Build k-d tree for neighbor search
        tree = cKDTree(points)

        normals = []
        for point in points:
            # Find k nearest neighbors
            _, indices = tree.query(point, k=min(k, len(points)))
            neighbor_points = points[indices]

            # Calculate covariance matrix
            cov_matrix = np.cov(neighbor_points.T)

            # Get eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Normal is the eigenvector corresponding to smallest eigenvalue
            normal = eigenvectors[:, 0]
            normals.append(normal)

        return np.array(normals)


# Example usage
def main():
    reconstructor = Simple3DReconstructor()

    # Simulate depth images from different viewpoints
    # In practice, these would come from an RGB-D camera

    # Create synthetic depth images (simplified)
    depth_images = []
    poses = []

    # Camera intrinsics (example values)
    camera_intrinsics = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 319.5,
        'cy': 239.5
    }

    # Simulate a few poses around an object
    for i in range(8):
        # Create a simple depth image (a plane at z=2)
        depth_img = np.full((480, 640), 2.0, dtype=np.float32)

        # Add some "objects" to the depth image
        depth_img[200:280, 300:340] = 1.5  # A box
        depth_img[150:170, 400:450] = 1.8  # A pole

        depth_images.append(depth_img)

        # Create a pose (rotation and translation)
        yaw = i * (2 * np.pi / 8)
        x = 3 * np.cos(yaw)
        y = 3 * np.sin(yaw)
        z = 1.0  # Height

        # Simple pose matrix (looking at origin)
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z

        # Add rotation to look at origin
        pose[:3, :3] = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        poses.append(pose)

    # Reconstruct 3D model
    print(f'Processing {len(depth_images)} depth images...')

    # Convert to point clouds and integrate
    all_points = reconstructor.integrate_depth_images(depth_images, poses, camera_intrinsics)

    print(f'Reconstructed {len(all_points)} points')

    # Downsample for visualization
    downsampled = reconstructor.voxel_grid_downsample(all_points, voxel_size=0.05)
    print(f'Downsampled to {len(downsampled)} points')


if __name__ == '__main__':
    main()
```

## Scene Understanding

### Object Recognition in 3D

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor


class SceneUnderstanding:
    def __init__(self):
        self.object_database = {}  # Store known object models

    def register_object_model(self, name, point_cloud, features=None):
        """Register a known object model"""
        self.object_database[name] = {
            'model': point_cloud,
            'features': features or self.extract_features(point_cloud)
        }

    def extract_features(self, point_cloud):
        """Extract geometric features from point cloud"""
        features = {}

        # Basic statistics
        features['centroid'] = np.mean(point_cloud, axis=0)
        features['size'] = np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0)
        features['volume'] = np.prod(features['size'])
        features['num_points'] = len(point_cloud)

        # Principal component analysis
        centered = point_cloud - features['centroid']
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        features['pca_axes'] = eigenvectors
        features['pca_values'] = eigenvalues

        # Calculate shape descriptors
        features['linearity'] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        features['planarity'] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        features['scattering'] = eigenvalues[2] / eigenvalues[0]

        return features

    def segment_objects(self, scene_points, distance_threshold=0.3, min_points=20):
        """Segment objects in scene using clustering"""
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=distance_threshold, min_samples=min_points)
        labels = clustering.fit_predict(scene_points[:, :3])

        objects = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue

            # Get points belonging to this cluster
            object_points = scene_points[labels == label]

            # Extract features for this object
            features = self.extract_features(object_points)

            objects.append({
                'points': object_points,
                'features': features,
                'label': label
            })

        return objects

    def match_object(self, object_features, threshold=0.1):
        """Match object features to known models"""
        best_match = None
        best_score = float('inf')

        for name, model in self.object_database.items():
            # Calculate similarity score (simplified - in practice, use more sophisticated methods)
            score = self.calculate_feature_similarity(object_features, model['features'])

            if score < best_score and score < threshold:
                best_score = score
                best_match = name

        return best_match, best_score

    def calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two sets of features"""
        # Simplified similarity calculation
        # In practice, you'd use more sophisticated methods like ICP, feature matching, etc.

        # Compare centroids
        centroid_diff = np.linalg.norm(features1['centroid'] - features2['centroid'])

        # Compare sizes
        size_diff = np.linalg.norm(features1['size'] - features2['size'])

        # Compare PCA values
        pca_diff = np.linalg.norm(features1['pca_values'] - features2['pca_values'])

        # Combine differences (weighted)
        similarity = 0.5 * centroid_diff + 0.3 * size_diff + 0.2 * pca_diff

        return similarity

    def understand_scene(self, scene_points):
        """Perform complete scene understanding"""
        # 1. Segment objects
        objects = self.segment_objects(scene_points)

        # 2. Classify each object
        scene_description = []
        for obj in objects:
            # Match to known objects
            match_name, confidence = self.match_object(obj['features'])

            scene_element = {
                'type': match_name if match_name else 'unknown_object',
                'confidence': confidence,
                'features': obj['features'],
                'points': obj['points']
            }

            scene_description.append(scene_element)

        return scene_description


# Example usage
def main():
    # Create scene understanding instance
    scene_understanding = SceneUnderstanding()

    # Register some known object models
    # Cube model (simplified)
    cube_points = []
    for x in np.linspace(-0.1, 0.1, 10):
        for y in np.linspace(-0.1, 0.1, 10):
            for z in [0.1, -0.1]:  # Top and bottom
                cube_points.append([x, y, z])
            for z in np.linspace(-0.1, 0.1, 10):
                for side_x, side_y in [(-0.1, y), (0.1, y), (x, -0.1), (x, 0.1)]:
                    cube_points.append([side_x, side_y, z])

    scene_understanding.register_object_model('cube', np.array(cube_points))

    # Cylinder model (simplified)
    cylinder_points = []
    for height in np.linspace(-0.1, 0.1, 10):
        for angle in np.linspace(0, 2*np.pi, 20):
            x = 0.1 * np.cos(angle)
            y = 0.1 * np.sin(angle)
            z = height
            cylinder_points.append([x, y, z])

    scene_understanding.register_object_model('cylinder', np.array(cylinder_points))

    # Create a scene with objects
    scene_points = []

    # Add a cube at (1, 1, 0)
    cube_offset = np.array([1, 1, 0])
    for point in cube_points:
        scene_points.append(np.array(point) + cube_offset)

    # Add a cylinder at (2, 0, 0)
    cylinder_offset = np.array([2, 0, 0])
    for point in cylinder_points:
        scene_points.append(np.array(point) + cylinder_offset)

    # Add some random points (clutter)
    random_points = np.random.uniform(-3, 3, (100, 3))
    random_points[:, 2] = np.abs(random_points[:, 2]) * 0.1  # Keep near ground
    for point in random_points:
        scene_points.append(point)

    scene_points = np.array(scene_points)

    print(f'Created scene with {len(scene_points)} points')

    # Perform scene understanding
    scene_description = scene_understanding.understand_scene(scene_points)

    print(f'\nScene understanding results:')
    for i, element in enumerate(scene_description):
        print(f'Object {i+1}: {element["type"]} (confidence: {element["confidence"]:.3f})')
        print(f'  Position: ({element["features"]["centroid"][0]:.2f}, {element["features"]["centroid"][1]:.2f}, {element["features"]["centroid"][2]:.2f})')
        print(f'  Size: ({element["features"]["size"][0]:.2f}, {element["features"]["size"][1]:.2f}, {element["features"]["size"][2]:.2f})')


if __name__ == '__main__':
    main()
```

## ROS 2 3D Perception Pipeline

### Complete 3D Perception Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor


class Perception3DNode(Node):
    def __init__(self):
        super().__init__('perception_3d_node')

        # Subscribe to point cloud data
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/points_raw',
            self.pointcloud_callback,
            10)

        # Subscribe to laser scan (for occupancy grid)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Publishers
        self.cluster_pub = self.create_publisher(MarkerArray, '/object_clusters', 10)
        self.ground_pub = self.create_publisher(MarkerArray, '/ground_plane', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/occupancy_map', 10)

        # Parameters
        self.voxel_size = 0.1
        self.cluster_eps = 0.3
        self.cluster_min_points = 20
        self.ground_threshold = 0.1

        # State
        self.occupancy_grid = {}
        self.robot_pose = (0.0, 0.0, 0.0)  # x, y, theta

        self.get_logger().info('3D Perception node initialized')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud"""
        try:
            # Convert to numpy array
            points = list(point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            ))
            points = np.array(points)

            if len(points) == 0:
                return

            # Filter ground plane
            ground_points, obstacle_points = self.segment_ground_plane(points)

            # Cluster obstacles
            clusters = self.cluster_points(obstacle_points)

            # Publish results
            self.publish_clusters(clusters, obstacle_points)
            self.publish_ground_plane(ground_points)

        except Exception as e:
            self.get_logger().error(f'Error in point cloud callback: {e}')

    def scan_callback(self, msg):
        """Process laser scan for occupancy mapping"""
        try:
            # Convert scan to points
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            valid_ranges = np.array(msg.ranges)
            valid_angles = angles[np.isfinite(valid_ranges)]
            valid_ranges = valid_ranges[np.isfinite(valid_ranges)]

            # Convert to Cartesian coordinates
            x_points = valid_ranges * np.cos(valid_angles)
            y_points = valid_ranges * np.sin(valid_angles)
            scan_points = np.column_stack([x_points, y_points, np.zeros_like(x_points)])

            # Update occupancy grid
            self.update_occupancy_grid(scan_points, self.robot_pose)

            # Publish map
            self.publish_occupancy_map()

        except Exception as e:
            self.get_logger().error(f'Error in scan callback: {e}')

    def segment_ground_plane(self, points):
        """Segment ground plane using RANSAC"""
        if len(points) < 10:
            return np.array([]), points

        # Use only x, y, z coordinates
        coords = points[:, :3]

        # Apply RANSAC to find ground plane
        ransac = RANSACRegressor(
            min_samples=3,
            residual_threshold=self.ground_threshold,
            max_trials=1000
        )

        # Try to fit z = ax + by + c (or z = constant for flat ground)
        X = coords[:, [0, 1]]  # x, y
        y = coords[:, 2]       # z

        try:
            ransac.fit(X, y)
            ground_mask = np.abs(ransac.predict(X) - y) < self.ground_threshold
            ground_points = points[ground_mask]
            obstacle_points = points[~ground_mask]
        except:
            # If RANSAC fails, use simple height thresholding
            z_median = np.median(coords[:, 2])
            ground_mask = np.abs(coords[:, 2] - z_median) < self.ground_threshold
            ground_points = points[ground_mask]
            obstacle_points = points[~ground_mask]

        return ground_points, obstacle_points

    def cluster_points(self, points):
        """Cluster points using DBSCAN"""
        if len(points) < self.cluster_min_points:
            return []

        # Use x, y coordinates for clustering (ignore z)
        coords = points[:, :2]

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_points)
        labels = clustering.fit_predict(coords)

        clusters = []
        for label in set(labels):
            if label == -1:  # Noise
                continue

            cluster_points = points[labels == label]
            clusters.append(cluster_points)

        return clusters

    def publish_clusters(self, clusters, all_points):
        """Publish clusters as visualization markers"""
        marker_array = MarkerArray()

        for i, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "clusters"
            marker.id = i
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD

            # Set scale
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Set color
            marker.color.r = float(i % 3) / 2.0
            marker.color.g = float((i + 1) % 3) / 2.0
            marker.color.b = float((i + 2) % 3) / 2.0
            marker.color.a = 0.8

            # Add points
            for point in cluster:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)

            marker_array.markers.append(marker)

        self.cluster_pub.publish(marker_array)

    def publish_ground_plane(self, ground_points):
        """Publish ground plane as visualization marker"""
        if len(ground_points) == 0:
            return

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ground"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set scale
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # Set color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.6

        # Add points
        for point in ground_points[:500]:  # Limit number of points for performance
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)

        ground_marker = MarkerArray()
        ground_marker.markers.append(marker)
        self.ground_pub.publish(ground_marker)

    def update_occupancy_grid(self, scan_points, robot_pose):
        """Update occupancy grid with scan data"""
        # Simplified occupancy grid update
        resolution = 0.2
        grid_size = 200  # 200x200 grid, 40mx40m area

        # Convert to grid coordinates
        grid_x = np.floor((scan_points[:, 0] - robot_pose[0]) / resolution).astype(int)
        grid_y = np.floor((scan_points[:, 1] - robot_pose[1]) / resolution).astype(int)

        # Update grid (simplified)
        valid_mask = (grid_x >= -grid_size//2) & (grid_x < grid_size//2) & \
                     (grid_y >= -grid_size//2) & (grid_y < grid_size//2)

        for x, y in zip(grid_x[valid_mask], grid_y[valid_mask]):
            grid_key = (int(x), int(y))
            self.occupancy_grid[grid_key] = 0.9  # Occupied

    def publish_occupancy_map(self):
        """Publish occupancy map as markers"""
        marker_array = MarkerArray()

        resolution = 0.2
        for i, ((grid_x, grid_y), occupancy) in enumerate(list(self.occupancy_grid.items())[:1000]):  # Limit for performance
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "map"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set position
            marker.pose.position.x = grid_x * resolution
            marker.pose.position.y = grid_y * resolution
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            # Set scale
            marker.scale.x = resolution
            marker.scale.y = resolution
            marker.scale.z = 0.1

            # Set color based on occupancy
            marker.color.r = occupancy
            marker.color.g = 1.0 - occupancy
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.map_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    perception_3d_node = Perception3DNode()

    try:
        rclpy.spin(perception_3d_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_3d_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: 3D Scene Understanding System

### Objective
Create a complete 3D scene understanding system that processes point cloud data, segments objects, and builds an occupancy map.

### Prerequisites
- Completed Chapter 1-5
- ROS 2 Humble with Gazebo installed
- Basic understanding of 3D perception concepts

### Steps

1. **Create a 3D perception package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python perception_3d_lab --dependencies rclpy sensor_msgs visualization_msgs geometry_msgs cv_bridge opencv-python numpy sklearn scipy
   ```

2. **Create the main perception node** (`perception_3d_lab/perception_3d_lab/perception_3d_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import PointCloud2, LaserScan
   from sensor_msgs_py import point_cloud2
   from visualization_msgs.msg import MarkerArray, Marker
   from geometry_msgs.msg import Point
   from std_msgs.msg import ColorRGBA, Bool
   from nav_msgs.msg import OccupancyGrid
   import numpy as np
   from sklearn.cluster import DBSCAN
   from sklearn.linear_model import RANSACRegressor
   import time
   import math


   class Perception3DLabNode(Node):
       def __init__(self):
           super().__init__('perception_3d_lab_node')

           # Subscribe to sensor data
           self.pc_sub = self.create_subscription(
               PointCloud2,
               '/camera/depth/color/points',  # Typical topic for RGB-D point cloud
               self.pointcloud_callback,
               10)
           self.scan_sub = self.create_subscription(
               LaserScan,
               '/scan',
               self.scan_callback,
               10)

           # Publishers
           self.cluster_pub = self.create_publisher(MarkerArray, '/object_clusters', 10)
           self.map_pub = self.create_publisher(OccupancyGrid, '/local_map', 10)
           self.status_pub = self.create_publisher(Bool, '/perception_ready', 10)

           # Parameters
           self.voxel_size = 0.05
           self.cluster_eps = 0.2
           self.cluster_min_points = 10
           self.ground_threshold = 0.1
           self.map_resolution = 0.1
           self.map_width = 400  # 40m x 40m map
           self.map_height = 400

           # State
           self.occupancy_map = np.full((self.map_height, self.map_width), 0.5, dtype=np.float32)  # 0.5 = unknown
           self.map_origin_x = -self.map_width * self.map_resolution / 2
           self.map_origin_y = -self.map_height * self.map_resolution / 2
           self.robot_x = 0.0
           self.robot_y = 0.0
           self.last_process_time = time.time()

           # Timer for periodic processing
           self.process_timer = self.create_timer(0.5, self.process_timer_callback)

           self.get_logger().info('3D Perception Lab node initialized')

       def pointcloud_callback(self, msg):
           """Process incoming point cloud"""
           try:
               # Convert to numpy array
               points = list(point_cloud2.read_points(
                   msg,
                   field_names=("x", "y", "z"),
                   skip_nans=True
               ))
               points = np.array(points)

               if len(points) == 0:
                   return

               # Downsample point cloud
               downsampled = self.voxel_grid_filter(points, self.voxel_size)

               # Segment ground plane
               ground_points, obstacle_points = self.segment_ground_plane(downsampled)

               # Cluster obstacles
               clusters = self.cluster_points(obstacle_points)

               # Update occupancy map with obstacle information
               self.update_occupancy_map(clusters)

               # Publish results
               self.publish_clusters(clusters)
               self.publish_occupancy_grid()

           except Exception as e:
               self.get_logger().error(f'Error processing point cloud: {e}')

       def scan_callback(self, msg):
           """Process laser scan to update map"""
           try:
               # Convert scan to obstacle points
               angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
               valid_ranges = np.array(msg.ranges)
               valid_mask = np.isfinite(valid_ranges) & (valid_ranges > 0) & (valid_ranges < msg.range_max)

               if not np.any(valid_mask):
                   return

               ranges = valid_ranges[valid_mask]
               angles = angles[valid_mask]

               # Convert to Cartesian coordinates
               x_scan = ranges * np.cos(angles)
               y_scan = ranges * np.sin(angles)
               scan_points = np.column_stack([x_scan, y_scan, np.zeros_like(x_scan)])

               # Update occupancy map
               self.update_map_with_scan(scan_points)

           except Exception as e:
               self.get_logger().error(f'Error processing scan: {e}')

       def voxel_grid_filter(self, points, voxel_size):
           """Downsample point cloud using voxel grid"""
           if len(points) == 0:
               return points

           # Calculate voxel coordinates
           voxel_coords = np.floor(points[:, :3] / voxel_size).astype(int)

           # Create unique voxel keys
           unique_voxels, indices = np.unique(voxel_coords, axis=0, return_index=True)

           return points[indices]

       def segment_ground_plane(self, points):
           """Segment ground plane using RANSAC"""
           if len(points) < 10:
               return np.array([]), points

           # Use x, y, z coordinates
           coords = points[:, :3]

           # Prepare data for RANSAC (z = ax + by + c)
           X = coords[:, [0, 1]]  # x, y
           y = coords[:, 2]       # z

           ransac = RANSACRegressor(
               min_samples=3,
               residual_threshold=self.ground_threshold,
               max_trials=1000
           )

           try:
               ransac.fit(X, y)
               ground_mask = np.abs(ransac.predict(X) - y) < self.ground_threshold
               ground_points = points[ground_mask]
               obstacle_points = points[~ground_mask]
           except:
               # Fallback to simple height thresholding
               z_median = np.median(coords[:, 2])
               ground_mask = np.abs(coords[:, 2] - z_median) < self.ground_threshold
               ground_points = points[ground_mask]
               obstacle_points = points[~ground_mask]

           return ground_points, obstacle_points

       def cluster_points(self, points):
           """Cluster points using DBSCAN"""
           if len(points) < self.cluster_min_points:
               return []

           # Use x, y coordinates for clustering
           coords = points[:, :2]

           clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_points)
           labels = clustering.fit_predict(coords)

           clusters = []
           for label in set(labels):
               if label == -1:  # Noise
                   continue

               cluster_points = points[labels == label]
               clusters.append(cluster_points)

           return clusters

       def update_occupancy_map(self, clusters):
           """Update occupancy map with clustered objects"""
           for cluster in clusters:
               if len(cluster) == 0:
                   continue

               # Calculate cluster centroid
               centroid = np.mean(cluster[:, :2], axis=0)

               # Convert to map coordinates
               map_x = int((centroid[0] - self.map_origin_x) / self.map_resolution)
               map_y = int((centroid[1] - self.map_origin_y) / self.map_resolution)

               # Check bounds
               if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                   # Mark as occupied
                   self.occupancy_map[map_y, map_x] = 0.9

       def update_map_with_scan(self, scan_points):
           """Update map with laser scan data"""
           for point in scan_points:
               # Convert to map coordinates
               map_x = int((point[0] - self.map_origin_x) / self.map_resolution)
               map_y = int((point[1] - self.map_origin_y) / self.map_resolution)

               # Check bounds
               if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                   # Mark as occupied (simplified - in practice, use proper ray tracing)
                   self.occupancy_map[map_y, map_x] = 0.8

       def publish_clusters(self, clusters):
           """Publish clusters as visualization markers"""
           marker_array = MarkerArray()

           for i, cluster in enumerate(clusters):
               if len(cluster) == 0:
                   continue

               marker = Marker()
               marker.header.frame_id = "base_link"
               marker.header.stamp = self.get_clock().now().to_msg()
               marker.ns = "objects"
               marker.id = i
               marker.type = Marker.SPHERE
               marker.action = Marker.ADD

               # Calculate bounding box for visualization
               min_pt = np.min(cluster[:, :3], axis=0)
               max_pt = np.max(cluster[:, :3], axis=0)
               center = (min_pt + max_pt) / 2

               # Set position
               marker.pose.position.x = center[0]
               marker.pose.position.y = center[1]
               marker.pose.position.z = center[2]
               marker.pose.orientation.w = 1.0

               # Set scale (bounding box size)
               marker.scale.x = max_pt[0] - min_pt[0]
               marker.scale.y = max_pt[1] - min_pt[1]
               marker.scale.z = max_pt[2] - min_pt[2]

               # Set color
               marker.color.r = 1.0
               marker.color.g = 0.0
               marker.color.b = 0.0
               marker.color.a = 0.7

               marker_array.markers.append(marker)

           self.cluster_pub.publish(marker_array)

       def publish_occupancy_grid(self):
           """Publish occupancy grid"""
           msg = OccupancyGrid()
           msg.header.frame_id = "map"
           msg.header.stamp = self.get_clock().now().to_msg()

           # Set metadata
           msg.info.resolution = self.map_resolution
           msg.info.width = self.map_width
           msg.info.height = self.map_height
           msg.info.origin.position.x = self.map_origin_x
           msg.info.origin.position.y = self.map_origin_y
           msg.info.origin.position.z = 0.0
           msg.info.origin.orientation.w = 1.0

           # Convert probabilities to int8 format (-1: unknown, 0-100: occupied percentage)
           grid_data = (self.occupancy_map * 100).astype(np.int8)
           grid_data = np.clip(grid_data, 0, 100)  # Ensure values are in range
           grid_data = np.where(self.occupancy_map < 0.2, 0, grid_data)  # Free space
           grid_data = np.where(self.occupancy_map > 0.8, 100, grid_data)  # Occupied space
           grid_data = np.where((self.occupancy_map >= 0.2) & (self.occupancy_map <= 0.8), -1, grid_data)  # Unknown

           # Flatten and convert to list
           msg.data = grid_data.flatten().tolist()

           self.map_pub.publish(msg)

       def process_timer_callback(self):
           """Periodic processing"""
           # Publish status
           status_msg = Bool()
           status_msg.data = True
           self.status_pub.publish(status_msg)

           # Log map statistics periodically
           if time.time() - self.last_process_time > 5.0:  # Every 5 seconds
               occupied_cells = np.sum(self.occupancy_map > 0.7)
               free_cells = np.sum(self.occupancy_map < 0.3)
               total_cells = self.occupancy_map.size

               self.get_logger().info(
                   f'Map: {occupied_cells} occupied, {free_cells} free, '
                   f'{total_cells - occupied_cells - free_cells} unknown cells'
               )
               self.last_process_time = time.time()


   def main(args=None):
       rclpy.init(args=args)
       perception_3d_lab_node = Perception3DLabNode()

       try:
           rclpy.spin(perception_3d_lab_node)
       except KeyboardInterrupt:
           pass
       finally:
           perception_3d_lab_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`perception_3d_lab/launch/perception_3d_lab.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='true',
           description='Use simulation (Gazebo) clock if true'
       )

       # Perception 3D node
       perception_3d_node = Node(
           package='perception_3d_lab',
           executable='perception_3d_node',
           name='perception_3d_lab_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           perception_3d_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'perception_3d_lab'

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
       description='3D Perception lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'perception_3d_node = perception_3d_lab.perception_3d_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select perception_3d_lab
   source install/setup.bash
   ```

6. **Run the 3D perception system**:
   ```bash
   ros2 launch perception_3d_lab perception_3d_lab.launch.py
   ```

### Expected Results
- The system should process point cloud data and identify objects
- Clusters should be published as visualization markers
- An occupancy map should be generated and published
- The system should integrate both 3D point cloud and 2D laser scan data

### Troubleshooting Tips
- Ensure your robot has both 3D and 2D sensors publishing data
- Check topic names match your robot's sensor configuration
- Adjust clustering parameters based on your environment
- Verify TF frames are properly configured

## Summary

In this chapter, we've explored the fundamental concepts of 3D perception and scene understanding, including point cloud processing, segmentation, mapping, and object recognition. We've implemented practical examples of each concept and created a complete 3D scene understanding system.

The hands-on lab provided experience with creating a system that processes both 3D point clouds and 2D laser scans to build a comprehensive understanding of the environment. This foundation is essential for more advanced robotic capabilities like navigation, manipulation, and interaction with the environment, which we'll explore in the upcoming chapters.