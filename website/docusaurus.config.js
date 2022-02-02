const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'spock',
  tagline: 'Managing complex configurations any other way would be highly illogical...',
  url: 'https://fidelity.github.io',
  baseUrl: '/spock/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/logo_small.png',
  organizationName: 'fidelity',
  projectName: 'spock',
  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/fidelity/spock/edit/master/website/',
          routeBasePath: '/'
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      announcementBar: {
        id: 'star-me',
        content: 'If you find spock useful give us a ⭐️ on our <a target="_blank" rel="noopener noreferrer" href="https://github.com/fidelity/spock">Github</a> repo'
      },
      navbar: {
        // title: 'spock',
        logo: {
          alt: 'MosaicML Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'index',
            position: 'left',
            label: 'Docs',
          },
          // {
          //   type: 'doc',
          //   docId: 'api',
          //   position: 'left',
          //   label: 'API'
          // },
          {
            href: 'https://github.com/fidelity/spock',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Home',
                to: '/',
              },
              {
                label: 'Quick Start',
                to: 'Quick-Start',
              },
              {
                label: 'Examples',
                href: 'https://github.com/fidelity/spock/blob/master/examples',
              },
              {
                label: 'API',
                to: '/api'
              },
            ],
          },
          {
            title: 'Maintainers',
            items: [
              {
                label: 'ncilfone',
                href: 'https://github.com/ncilfone',
              },
            ],
          },
          {
            title: 'Open Source @ Fidelity',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/fidelity',
              },
            ],
          },
        ],
        copyright: `Copyright © <a href="mailto: opensource@fidelity.com">FMR LLC</a>. Built with <a href="https://docusaurus.io/">Docusaurus</a>`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
});
