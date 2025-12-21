// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Part I: Foundations of Physical AI and Robotics',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Introduction to Physical AI & Embodied Intelligence',
          items: [
            'part-i-foundations/chapter-1-introduction/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Robot Operating System (ROS 2) Fundamentals',
          items: [
            'part-i-foundations/chapter-2-ros-fundamentals/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Robot Modeling and Simulation Fundamentals',
          items: [
            'part-i-foundations/chapter-3-robot-modeling/index',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Part II: Perception and Understanding',
      items: [
        {
          type: 'category',
          label: 'Chapter 4: Sensor Integration and Data Processing',
          items: [
            'part-ii-perception/chapter-4-sensor-integration/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 5: Computer Vision for Robotics',
          items: [
            'part-ii-perception/chapter-5-computer-vision/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 6: 3D Perception and Scene Understanding',
          items: [
            'part-ii-perception/chapter-6-3d-perception/index',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Part III: Motion and Control',
      items: [
        {
          type: 'category',
          label: 'Chapter 7: Kinematics and Dynamics',
          items: [
            'part-iii-motion/chapter-7-kinematics/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 8: Locomotion and Balance Control',
          items: [
            'part-iii-motion/chapter-8-locomotion/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 9: Motion Planning and Navigation',
          items: [
            'part-iii-motion/chapter-9-navigation/index',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Part IV: Intelligence and Learning',
      items: [
        {
          type: 'category',
          label: 'Chapter 10: Reinforcement Learning for Robotics',
          items: [
            'part-iv-intelligence/chapter-10-reinforcement-learning/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 11: Imitation Learning and VLA',
          items: [
            'part-iv-intelligence/chapter-11-imitation-learning/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 12: Human-Robot Interaction',
          items: [
            'part-iv-intelligence/chapter-12-hri/index',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Part V: Integration and Applications',
      items: [
        {
          type: 'category',
          label: 'Chapter 13: Multi-Robot Systems and Coordination',
          items: [
            'part-v-integration/chapter-13-multi-robot/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 14: Real-World Deployment and Safety',
          items: [
            'part-v-integration/chapter-14-deployment/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 15: Advanced Topics and Future Directions',
          items: [
            'part-v-integration/chapter-15-future/index',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'VLA Book Series',
      items: [
        {
          type: 'category',
          label: 'Introduction to Physical AI & Humanoid Robotics',
          items: [
            'intro',
          ],
        },
        {
          type: 'category',
          label: 'ROS 2 Fundamentals',
          items: [
            'ros2',
          ],
        },
        {
          type: 'category',
          label: 'Robot Simulation (Gazebo & Unity)',
          items: [
            'gazebo',
          ],
        },
        {
          type: 'category',
          label: 'NVIDIA Isaac AI Brain',
          items: [
            'isaac',
          ],
        },
        {
          type: 'category',
          label: 'Vision-Language-Action (VLA)',
          items: [
            'vla',
          ],
        },
        {
          type: 'category',
          label: 'Capstone Project: Autonomous Humanoid',
          items: [
            'capstone',
          ],
        },
      ],
    },
  ],
};

module.exports = sidebars;