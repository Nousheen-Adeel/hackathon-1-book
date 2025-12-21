// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to embodied intelligence and humanoid robotics',
  favicon: 'img/logo.svg',

  // Set the production url of your site here
  url: 'https://your-book-url.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'your-username', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotics', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/your-username/physical-ai-humanoid-robotics/tree/main/',
        },
        blog: false, // Disable blog for this book
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themes: [
    // ... Your other themes
    '@docusaurus/theme-live-codeblock',
  ],

  // Custom fields for our application
  customFields: {
    backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',
  },
  
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book Chapters',
          },
          {
            type: 'dropdown',
            label: 'Resources',
            position: 'left',
            items: [
              {
                label: 'ROS 2 Tutorials',
                to: '/docs/ros2',
              },
              {
                label: 'Gazebo Simulation',
                to: '/docs/gazebo',
              },
              {
                label: 'NVIDIA Isaac',
                to: '/docs/isaac',
              },
              {
                label: 'VLA Models',
                to: '/docs/vla',
              },
            ],
          },
          {
            type: 'dropdown',
            label: 'Parts',
            position: 'left',
            items: [
              {
                label: 'Part I: Foundations',
                to: '/docs/part-i-foundations/chapter-1-introduction/index',
              },
              {
                label: 'Part II: Perception',
                to: '/docs/part-ii-perception/chapter-4-sensor-integration/index',
              },
              {
                label: 'Part III: Motion & Control',
                to: '/docs/part-iii-motion/chapter-7-kinematics/index',
              },
              {
                label: 'Part IV: Intelligence',
                to: '/docs/part-iv-intelligence/chapter-10-reinforcement-learning/index',
              },
              {
                label: 'Part V: Integration',
                to: '/docs/part-v-integration/chapter-13-multi-robot/index',
              },
            ],
          },
          {
            href: 'https://github.com/your-username/physical-ai-humanoid-robotics',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: '/docs/capstone',
            label: 'Capstone Project',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Book',
            items: [
              {
                label: 'Introduction',
                to: '/docs/part-i-foundations/chapter-1-introduction/index',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-username/physical-ai-humanoid-robotics',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['python', 'bash', 'json', 'yaml', 'cpp'],
      },
    }),
};

// Add configuration for the backend URL
config.customFields = {
  backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',
};

module.exports = config;