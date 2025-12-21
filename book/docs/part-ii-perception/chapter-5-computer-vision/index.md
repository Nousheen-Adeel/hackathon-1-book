---
title: Chapter 5 - Computer Vision for Robotics
sidebar_position: 2
---

# Chapter 5: Computer Vision for Robotics

## Learning Goals

- Apply computer vision techniques to robotic perception
- Understand visual SLAM and object recognition
- Learn real-time image processing
- Implement object detection and tracking
- Create visual SLAM pipeline
- Integrate vision with robot control

## Introduction to Computer Vision in Robotics

Computer vision is a critical component of robotic perception, enabling robots to interpret and understand visual information from their environment. Unlike traditional computer vision applications that process images in isolation, robotics applications require real-time processing, robustness to changing conditions, and integration with other sensors and control systems.

### Key Challenges in Robotic Vision

1. **Real-time Processing**: Robots must process visual information quickly to make timely decisions
2. **Dynamic Environments**: Lighting, viewpoints, and scenes constantly change
3. **Motion Blur**: Robot movement can cause image blur
4. **Computational Constraints**: Limited processing power on mobile robots
5. **Integration**: Vision must work seamlessly with other sensors and control systems

### ROS 2 Vision Ecosystem

ROS 2 provides several packages for computer vision:

- **vision_opencv**: Bridges between ROS 2 and OpenCV
- **image_transport**: Efficient image message transport
- **cv_bridge**: Conversions between ROS 2 and OpenCV formats
- **image_pipeline**: Collection of image processing tools
- **vision_msgs**: Standard message types for vision results

## Image Processing Fundamentals

### Basic Image Operations

```python
# Basic image processing with OpenCV in ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Basic image operations
        processed_image = self.process_image(cv_image)

        # Display results
        cv2.imshow('Original', cv_image)
        cv2.imshow('Processed', processed_image)
        cv2.waitKey(1)

    def process_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Combine results (for visualization)
        result = image.copy()
        result[edges > 0] = [0, 255, 0]  # Mark edges in green

        return result


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    cv2.destroyAllWindows()
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Image Filtering and Enhancement

```python
import cv2
import numpy as np


class ImageEnhancer:
    def __init__(self):
        pass

    def enhance_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """Enhance brightness and contrast of an image"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def apply_morphological_operations(self, image, kernel_size=5):
        """Apply morphological operations for noise reduction"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Erosion and dilation for noise removal
        erosion = cv2.erode(image, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # Opening and closing
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return closing

    def adaptive_threshold(self, image):
        """Apply adaptive thresholding for varying lighting conditions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def histogram_equalization(self, image):
        """Apply histogram equalization for better contrast"""
        if len(image.shape) == 3:
            # Convert to YUV for better histogram equalization
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)


# Example usage
def main():
    enhancer = ImageEnhancer()

    # Load an image (in practice, this would come from a camera)
    # image = cv2.imread('example.jpg')

    # For this example, create a synthetic image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Apply enhancements
    enhanced = enhancer.enhance_brightness_contrast(image, brightness=30, contrast=1.2)
    morphed = enhancer.apply_morphological_operations(enhanced)
    adaptive_thresh = enhancer.adaptive_threshold(morphed)
    hist_eq = enhancer.histogram_equalization(image)

    # Display results (in practice, you'd publish these as ROS messages)
    cv2.imshow('Original', image)
    cv2.imshow('Enhanced', enhanced)
    cv2.imshow('Morphed', morphed)
    cv2.imshow('Adaptive Threshold', adaptive_thresh)
    cv2.imshow('Histogram Equalized', hist_eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

## Feature Detection and Matching

### Corner Detection

```python
import cv2
import numpy as np


class FeatureDetector:
    def __init__(self):
        pass

    def detect_corners_harris(self, image, block_size=2, ksize=3, k=0.04):
        """Detect corners using Harris corner detector"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, block_size, ksize, k)

        # Result is dilated for marking the corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value
        corner_image = image.copy()
        corner_image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

        return corner_image, dst

    def detect_corners_shi_tomasi(self, image, max_corners=100, quality_level=0.01, min_distance=10):
        """Detect corners using Shi-Tomasi corner detector"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)

        corner_image = image.copy()
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(corner_image, (x, y), 3, [0, 0, 255], -1)

        return corner_image, corners

    def detect_sift_features(self, image):
        """Detect SIFT features (if available)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Create SIFT detector
            sift = cv2.SIFT_create()

            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            # Draw keypoints
            feature_image = cv2.drawKeypoints(image, keypoints, None)

            return feature_image, keypoints, descriptors
        except AttributeError:
            # SIFT may not be available in some OpenCV builds
            print("SIFT not available, using ORB instead")
            return self.detect_orb_features(image)

    def detect_orb_features(self, image):
        """Detect ORB features (free alternative to SIFT)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Create ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # Draw keypoints
        feature_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

        return feature_image, keypoints, descriptors


# Example usage
def main():
    detector = FeatureDetector()

    # Load an image (in practice, this would come from a camera)
    # image = cv2.imread('example.jpg')

    # For this example, create a synthetic image with some features
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)
    cv2.line(image, (400, 100), (500, 200), (0, 0, 255), 5)

    # Detect features
    harris_result, _ = detector.detect_corners_harris(image)
    shi_tomasi_result, _ = detector.detect_corners_shi_tomasi(image)
    orb_result, _, _ = detector.detect_orb_features(image)

    # Display results
    cv2.imshow('Original', image)
    cv2.imshow('Harris Corners', harris_result)
    cv2.imshow('Shi-Tomasi Corners', shi_tomasi_result)
    cv2.imshow('ORB Features', orb_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

## Object Detection

### Traditional Object Detection Methods

```python
import cv2
import numpy as np


class ObjectDetector:
    def __init__(self):
        # Load Haar cascade for face detection (as an example)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces_haar(self, image):
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        result_image = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return result_image, faces

    def detect_colors(self, image, lower_color, upper_color):
        """Detect objects of specific color"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for the color range
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = image.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return result_image, contours

    def template_matching(self, image, template, threshold=0.8):
        """Find template in image using template matching"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Find locations where the matching result is above threshold
        locations = np.where(result >= threshold)

        h, w = template_gray.shape
        result_image = image.copy()

        # Draw rectangles around matches
        for pt in zip(*locations[::-1]):
            cv2.rectangle(result_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

        return result_image, locations, result


# Example usage
def main():
    detector = ObjectDetector()

    # Create a test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(image, (400, 100), (500, 200), (0, 0, 255), 2)  # Red rectangle outline

    # Detect blue objects (BGR format)
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    color_result, contours = detector.detect_colors(image, lower_blue, upper_blue)

    # Display results
    cv2.imshow('Original', image)
    cv2.imshow('Color Detection', color_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

### Deep Learning-based Object Detection

```python
# This is a conceptual example - actual implementation would require
# additional dependencies like PyTorch or TensorFlow
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np


class DeepObjectDetector(Node):
    def __init__(self):
        super().__init__('deep_object_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publisher for detection results
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)

        self.bridge = CvBridge()

        # Load a pre-trained model (conceptual - in practice, this would load an actual model)
        self.load_model()

        self.get_logger().info('Deep object detector initialized')

    def load_model(self):
        """Load pre-trained object detection model"""
        # In practice, this would load a model like YOLO, SSD, or Faster R-CNN
        # For this example, we'll use a placeholder
        self.model = None
        self.get_logger().info('Model loaded')

    def preprocess_image(self, image):
        """Preprocess image for the neural network"""
        # Resize image to model input size
        input_size = (416, 416)  # Example size for YOLO
        resized = cv2.resize(image, input_size)

        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0

        # Convert to NCHW format (batch, channels, height, width)
        # This would be done with actual deep learning frameworks
        return normalized

    def postprocess_detections(self, raw_detections, original_shape):
        """Convert raw model outputs to detection messages"""
        # This is a simplified example
        # In practice, this would decode bounding boxes, class probabilities, etc.

        detections = Detection2DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera_frame'  # Should come from image header

        # Example: create mock detections
        for i in range(2):  # Simulate 2 detections
            detection = Detection2D()

            # Set bounding box (in practice, these would come from the model)
            detection.bbox.center.x = original_shape[1] // 2
            detection.bbox.center.y = original_shape[0] // 2
            detection.bbox.size_x = 100
            detection.bbox.size_y = 100

            # Set detection results
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = 'object'
            hypothesis.hypothesis.score = 0.95
            detection.results.append(hypothesis)

            detections.detections.append(detection)

        return detections

    def image_callback(self, msg):
        """Process incoming image and detect objects"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image
            preprocessed = self.preprocess_image(cv_image)

            # Run inference (conceptual)
            # raw_detections = self.model(preprocessed)

            # For this example, we'll simulate detection results
            raw_detections = np.random.random((1, 100, 6))  # Simulated detection format

            # Postprocess detections
            detection_array = self.postprocess_detections(raw_detections, cv_image.shape)

            # Publish detections
            self.detection_pub.publish(detection_array)

            self.get_logger().info(f'Published {len(detection_array.detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    deep_object_detector = DeepObjectDetector()

    try:
        rclpy.spin(deep_object_detector)
    except KeyboardInterrupt:
        pass
    finally:
        deep_object_detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Visual SLAM (Simultaneous Localization and Mapping)

### Feature-based SLAM Concepts

```python
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class SimpleVisualSLAM:
    def __init__(self):
        # ORB detector for feature detection
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Store keyframes and their features
        self.keyframes = deque(maxlen=100)
        self.current_pose = np.eye(4)  # 4x4 identity matrix
        self.poses = []  # Store trajectory

        # Feature tracking
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Camera parameters (example values)
        self.fx = 525.0  # Focal length x
        self.fy = 525.0  # Focal length y
        self.cx = 319.5  # Principal point x
        self.cy = 239.5  # Principal point y

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def detect_features(self, image):
        """Detect features in the current image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, kp1, desc1, kp2, desc2):
        """Match features between two images"""
        if desc1 is None or desc2 is None:
            return [], []

        matches = self.bf_matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep only good matches
        good_matches = [m for m in matches if m.distance < 50]

        # Extract corresponding points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return pts1, pts2, good_matches

    def estimate_motion(self, pts1, pts2):
        """Estimate camera motion between two frames"""
        if len(pts1) >= 8:
            # Compute fundamental matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 4, 0.999)
            F = F/F[2,2]  # Normalize

            # Essential matrix
            E = self.camera_matrix.T @ F @ self.camera_matrix

            # Decompose essential matrix
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)

            # Create transformation matrix
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t.ravel()

            return transformation
        else:
            return np.eye(4)  # No motion if not enough points

    def process_frame(self, image):
        """Process a single frame for SLAM"""
        # Detect features in current frame
        current_kp, current_desc = self.detect_features(image)

        # If we have previous frame, compute motion
        if self.prev_keypoints is not None and self.prev_descriptors is not None:
            # Match features
            pts1, pts2, matches = self.match_features(
                self.prev_keypoints, self.prev_descriptors,
                current_kp, current_desc
            )

            if len(pts1) > 0:
                # Estimate motion
                motion = self.estimate_motion(pts1, pts2)

                # Update current pose
                self.current_pose = self.current_pose @ motion

                # Store the pose
                self.poses.append(self.current_pose.copy())

        # Update previous frame data
        self.prev_keypoints = current_kp
        self.prev_descriptors = current_desc

        return self.current_pose.copy()

    def visualize_trajectory(self):
        """Visualize the estimated trajectory"""
        if len(self.poses) == 0:
            return

        # Extract x, y positions from poses
        positions = []
        for pose in self.poses:
            positions.append([pose[0, 3], pose[1, 3]])  # x, y coordinates

        positions = np.array(positions)

        plt.figure(figsize=(10, 8))
        plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory', linewidth=2)
        plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start', zorder=5)
        plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End', zorder=5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Estimated Trajectory from Visual SLAM')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()


# Example usage
def main():
    slam = SimpleVisualSLAM()

    # Simulate processing a sequence of images
    # In practice, these would come from a camera
    for i in range(50):
        # Create a synthetic "image" with some features
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Add some synthetic features (squares)
        cv2.rectangle(image, (100 + i*2, 100 + i*1), (150 + i*2, 150 + i*1), (255, 255, 255), -1)
        cv2.rectangle(image, (300 - i*1, 200 + i*2), (350 - i*1, 250 + i*2), (255, 0, 0), -1)

        # Process the frame
        current_pose = slam.process_frame(image)

        print(f"Frame {i+1}: Position = ({current_pose[0, 3]:.2f}, {current_pose[1, 3]:.2f}, {current_pose[2, 3]:.2f})")

    # Visualize the trajectory
    slam.visualize_trajectory()


if __name__ == '__main__':
    main()
```

## Object Tracking

### Single Object Tracking

```python
import cv2
import numpy as np


class ObjectTracker:
    def __init__(self):
        # Different tracker types available in OpenCV
        self.tracker_types = [
            'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'
        ]
        self.tracker_type = 'CSRT'  # CSRT is generally the most accurate

    def create_tracker(self):
        """Create tracker based on selected type"""
        if self.tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif self.tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif self.tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif self.tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif self.tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif self.tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif self.tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()

        return tracker

    def track_object(self, video_source=0):
        """Track object in video stream"""
        cap = cv2.VideoCapture(video_source)

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Cannot read video stream")
            return

        # Select region to track (manually or automatically)
        bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Tracking")

        # Create tracker
        tracker = self.create_tracker()

        # Initialize tracker with first frame and bounding box
        ret = tracker.init(frame, bbox)

        while True:
            # Read a new frame
            ret, frame = cap.read()
            if not ret:
                break

            # Update tracker
            success, bbox = tracker.update(frame)

            # Draw bounding box
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Display frame
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage for tracking
def main():
    tracker = ObjectTracker()

    # For this example, we'll create a simple tracking function
    # that processes a single frame with a known object
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 150), (300, 250), (255, 255, 255), -1)  # White square

    # Initialize tracker
    tracker_instance = tracker.create_tracker()
    bbox = (200, 150, 100, 100)  # x, y, width, height

    # Initialize tracker with the bounding box
    tracker_instance.init(image, bbox)

    print("Tracker initialized with bounding box:", bbox)


if __name__ == '__main__':
    main()
```

## Vision-Based Control

### Image-Based Visual Servoing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisualServoingController(Node):
    def __init__(self):
        super().__init__('visual_servoing_controller')

        # Subscribe to camera image
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Publish velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.bridge = CvBridge()

        # Target object parameters
        self.target_x = None
        self.target_y = None
        self.target_area = None

        # Control parameters
        self.kp_linear = 0.005  # Proportional gain for linear velocity
        self.kp_angular = 0.01  # Proportional gain for angular velocity
        self.target_area_desired = 5000  # Desired area of target in pixels

        # Object detection parameters
        self.lower_color = np.array([20, 100, 100])  # Lower HSV for color detection
        self.upper_color = np.array([30, 255, 255])  # Upper HSV for color detection

        self.get_logger().info('Visual servoing controller initialized')

    def detect_target(self, image):
        """Detect target object in image"""
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for target color
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assuming it's our target)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 100:  # Filter out small contours
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    return cx, cy, area, largest_contour

        return None, None, None, None

    def compute_control(self, image_width, image_height, target_x, target_y, target_area):
        """Compute control commands based on target position"""
        cmd_vel = Twist()

        if target_x is not None and target_y is not None:
            # Calculate errors
            error_x = target_x - image_width / 2  # Horizontal error (for rotation)
            error_y = target_y - image_height / 2  # Vertical error (ignored in this example)
            area_error = target_area - self.target_area_desired  # Area error (for forward/backward)

            # Compute control commands
            cmd_vel.linear.x = -self.kp_linear * area_error  # Move forward/backward based on size
            cmd_vel.angular.z = -self.kp_angular * error_x   # Rotate based on horizontal position

            # Apply velocity limits
            cmd_vel.linear.x = max(-0.5, min(0.5, cmd_vel.linear.x))
            cmd_vel.angular.z = max(-1.0, min(1.0, cmd_vel.angular.z))

        return cmd_vel

    def image_callback(self, msg):
        """Process incoming image and compute control commands"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect target in image
            self.target_x, self.target_y, self.target_area, contour = self.detect_target(cv_image)

            # Draw target if found
            if contour is not None:
                cv2.drawContours(cv_image, [contour], -1, (0, 255, 0), 3)
                cv2.circle(cv_image, (int(self.target_x), int(self.target_y)), 5, (0, 0, 255), -1)

            # Compute control commands
            cmd_vel = self.compute_control(
                cv_image.shape[1], cv_image.shape[0],  # image width, height
                self.target_x, self.target_y, self.target_area
            )

            # Publish control commands
            self.cmd_vel_pub.publish(cmd_vel)

            # Log control information
            if self.target_x is not None:
                self.get_logger().info(
                    f'Target: pos=({self.target_x:.1f}, {self.target_y:.1f}), '
                    f'area={self.target_area:.1f}, '
                    f'cmd_vel=({cmd_vel.linear.x:.3f}, {cmd_vel.angular.z:.3f})'
                )

            # Display image with target
            cv2.imshow('Visual Servoing', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    visual_servoing_controller = VisualServoingController()

    try:
        rclpy.spin(visual_servoing_controller)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        visual_servoing_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Hands-On Lab: Vision-Based Navigation System

### Objective
Create a complete vision-based navigation system that detects colored objects and navigates toward them using visual servoing.

### Prerequisites
- Completed Chapter 1-4
- ROS 2 Humble with Gazebo installed
- Basic understanding of robot control

### Steps

1. **Create a vision navigation package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python vision_navigation_lab --dependencies rclpy sensor_msgs geometry_msgs cv_bridge opencv-python numpy
   ```

2. **Create the vision navigation node** (`vision_navigation_lab/vision_navigation_lab/vision_navigation_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from geometry_msgs.msg import Twist, Point
   from cv_bridge import CvBridge
   import cv2
   import numpy as np
   from std_msgs.msg import Bool
   import time


   class VisionNavigationNode(Node):
       def __init__(self):
           super().__init__('vision_navigation_node')

           # Subscribe to camera image
           self.image_sub = self.create_subscription(
               Image,
               '/camera/image_raw',
               self.image_callback,
               10)

           # Publish velocity commands
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           # Publish navigation status
           self.nav_status_pub = self.create_publisher(Bool, '/navigation_active', 10)

           # CV bridge for image processing
           self.bridge = CvBridge()

           # Navigation state
           self.target_detected = False
           self.navigation_active = False
           self.last_detection_time = time.time()
           self.detection_timeout = 3.0  # Stop if no detection for 3 seconds

           # Control parameters
           self.kp_linear = 0.003
           self.kp_angular = 0.01
           self.target_area_desired = 8000
           self.min_target_area = 500
           self.max_target_area = 50000

           # Target color in HSV (orange example)
           self.lower_color = np.array([10, 100, 100])
           self.upper_color = np.array([30, 255, 255])

           # Timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)

           self.get_logger().info('Vision navigation node initialized')

       def detect_target(self, image):
           """Detect colored target in image"""
           # Convert to HSV
           hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

           # Create mask for target color
           mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

           # Apply morphological operations
           kernel = np.ones((7, 7), np.uint8)
           mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
           mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

           # Find contours
           contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

           if contours:
               # Find largest contour
               largest_contour = max(contours, key=cv2.contourArea)
               area = cv2.contourArea(largest_contour)

               if area > self.min_target_area and area < self.max_target_area:
                   # Calculate centroid
                   M = cv2.moments(largest_contour)
                   if M["m00"] != 0:
                       cx = int(M["m10"] / M["m00"])
                       cy = int(M["m01"] / M["m00"])
                       return cx, cy, area, largest_contour

           return None, None, None, None

       def compute_navigation_command(self, image_width, image_height, target_x, target_y, target_area):
           """Compute navigation commands based on target position"""
           cmd_vel = Twist()

           if target_x is not None and target_y is not None:
               # Calculate errors
               error_x = target_x - image_width / 2
               area_error = target_area - self.target_area_desired

               # Compute velocities
               cmd_vel.linear.x = -self.kp_linear * area_error
               cmd_vel.angular.z = -self.kp_angular * error_x

               # Apply limits
               cmd_vel.linear.x = max(-0.4, min(0.4, cmd_vel.linear.x))
               cmd_vel.angular.z = max(-1.0, min(1.0, cmd_vel.angular.z))

               # Stop if close enough to target
               if abs(area_error) < 1000:  # If area is close to desired
                   cmd_vel.linear.x = 0.0
                   cmd_vel.angular.z = 0.0
                   self.get_logger().info('Target reached!')

           return cmd_vel

       def image_callback(self, msg):
           """Process incoming image"""
           try:
               # Convert to OpenCV
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

               # Detect target
               target_x, target_y, target_area, contour = self.detect_target(cv_image)

               # Update detection status
               if target_x is not None:
                   self.target_detected = True
                   self.last_detection_time = time.time()

                   # Draw target on image
                   if contour is not None:
                       cv2.drawContours(cv_image, [contour], -1, (0, 255, 0), 3)
                       cv2.circle(cv_image, (int(target_x), int(target_y)), 5, (0, 0, 255), -1)
                       cv2.putText(cv_image, f'Area: {target_area:.0f}',
                                 (int(target_x), int(target_y) - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               else:
                   self.target_detected = False

               # Show image
               cv2.imshow('Vision Navigation', cv_image)
               cv2.waitKey(1)

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def control_loop(self):
           """Main control loop"""
           # Check for detection timeout
           if time.time() - self.last_detection_time > self.detection_timeout:
               self.target_detected = False

           # Determine if navigation should be active
           self.navigation_active = self.target_detected

           cmd_vel = Twist()

           if self.navigation_active:
               # Get current image dimensions (in a real system, you'd store this from the last image)
               # For this example, we'll use a fixed size
               img_width, img_height = 640, 480

               # We need to access the last detected target info
               # In a real system, you'd store this from the image callback
               # For this example, we'll simulate the detection
               cmd_vel = self.compute_navigation_command(
                   img_width, img_height,
                   320, 240, 4000  # Simulated target position and area
               )
           else:
               # Stop if no target detected recently
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = 0.0
               if self.target_detected:  # Only log once when target is lost
                   self.get_logger().info('Target lost, stopping')

           # Publish command and status
           self.cmd_vel_pub.publish(cmd_vel)

           status_msg = Bool()
           status_msg.data = self.navigation_active
           self.nav_status_pub.publish(status_msg)


   def main(args=None):
       rclpy.init(args=args)
       vision_navigation_node = VisionNavigationNode()

       try:
           rclpy.spin(vision_navigation_node)
       except KeyboardInterrupt:
           pass
       finally:
           cv2.destroyAllWindows()
           vision_navigation_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`vision_navigation_lab/launch/vision_navigation.launch.py`):
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

       # Vision navigation node
       vision_navigation_node = Node(
           package='vision_navigation_lab',
           executable='vision_navigation_node',
           name='vision_navigation_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           vision_navigation_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'vision_navigation_lab'

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
       description='Vision navigation lab for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'vision_navigation_node = vision_navigation_lab.vision_navigation_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select vision_navigation_lab
   source install/setup.bash
   ```

6. **Run the vision navigation system**:
   ```bash
   ros2 launch vision_navigation_lab vision_navigation.launch.py
   ```

### Expected Results
- The system should detect colored objects in the camera feed
- The robot should navigate toward the detected object
- Visual feedback should be displayed showing the detected target
- The system should stop when it gets close to the target

### Troubleshooting Tips
- Adjust color thresholds based on your target object
- Verify camera topic name matches your robot's camera
- Check that the robot's base controller is properly configured
- Ensure proper lighting conditions for color detection

## Summary

In this chapter, we've explored the fundamental concepts of computer vision for robotics, including image processing, feature detection, object detection, visual SLAM, and vision-based control. We've implemented practical examples of each concept and created a complete vision-based navigation system.

The hands-on lab provided experience with creating a vision-based navigation system that detects objects and controls robot motion based on visual feedback. This foundation is essential for more advanced robotic perception and control systems that we'll explore in the upcoming chapters.