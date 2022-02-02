/**
Docs sidebars -- split Docs vs. API spec
 */

// module.exports = {
//     docs: [
//         {
//             type: 'doc',
//             label: 'Home',
//             id: 'index',
//         },
//         // {
//         //     type: 'doc',
//         //     label: 'Installation',
//         //     id: 'Installation'
//         // },
//         // {
//         //     type: 'doc',
//         //     label: 'Quick Start',
//         //     id: 'Quick-Start'
//         // },
//         // {
//         //     type: 'doc',
//         //     label: 'argparse Replacement',
//         //     id: 'ArgParser-Replacement'
//         // },
//         // {
//         //     type: 'category',
//         //     label: 'Basics',
//         //     items: [
//         //         {
//         //             type: 'doc',
//         //             label: 'Introduction',
//         //             id: 'basics/About'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Define',
//         //             id: 'basics/Define'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Building',
//         //             id: 'basics/Building'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Configuration Files',
//         //             id: 'basics/Configuration-Files'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Saving',
//         //             id: 'basics/Saving'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Running',
//         //             id: 'basics/Run'
//         //         },
//         //     ],
//         // },
//         // {
//         //     type: 'category',
//         //     label: 'Advanced Usage',
//         //     items: [
//         //         {
//         //             type: 'doc',
//         //             label: 'Introduction',
//         //             id: 'advanced_features/About'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Setting Defaults',
//         //             id: 'advanced_features/Defaults'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Optional Parameters',
//         //             id: 'advanced_features/Optional-Parameters'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Parameter Groups',
//         //             id: 'advanced_features/Parameter-Groups'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Class Inheritance',
//         //             id: 'advanced_features/Inheritance'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Advanced Types/Nested Definitions',
//         //             id: 'advanced_features/Advanced-Types'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Lazy Dependencies',
//         //             id: "advanced_features/Lazy-Features"
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Local Definitions',
//         //             id: 'advanced_features/Local-Definitions'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Specifying Config Path(s)',
//         //             id: 'advanced_features/Keyword-Configs'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'CMD Line Keyword Arguments',
//         //             id: 'advanced_features/Command-Line-Overrides'
//         //         },
//         //         {
//         //             type: 'doc',
//         //             label: 'Evolve',
//         //             id: 'advanced_features/Evolve'
//         //         },
//         //     ],
//         // },
//         // {
//         //     type: 'category',
//         //     label: 'Addon Functionality',
//         //     items: [
//         //         {
//         //           type: 'doc',
//         //           label: 'S3 URI(s)',
//         //           id: 'addons/S3'
//         //         },
//         //         {
//         //             type: 'category',
//         //             label: 'Hyperparameter Tuning',
//         //             items: [
//         //                 {
//         //                     type: 'doc',
//         //                     label: 'Introduction',
//         //                     id: 'addons/tuner/About'
//         //                 },
//         //                 {
//         //                     type: 'doc',
//         //                     label: 'Basics',
//         //                     id: 'addons/tuner/Basics'
//         //                 },
//         //                 {
//         //                     type: 'doc',
//         //                     label: 'Ax Backend',
//         //                     id: 'addons/tuner/Ax'
//         //                 },
//         //                 {
//         //                     type: 'doc',
//         //                     label: 'Optuna Backend',
//         //                     id: 'addons/tuner/Optuna'
//         //                 },
//         //                 {
//         //                     type: 'doc',
//         //                     label: 'Saving',
//         //                     id: 'addons/tuner/Saving'
//         //                 },
//         //             ],
//         //         }
//         //     ],
//         // },
//         // {
//         //     type: 'doc',
//         //     label: 'Contributing',
//         //     id: 'contributing'
//         // },
//     ],
//     api: [
//         'api',
//         {
//             type: 'category',
//             label: 'Python API',
//             items: [
//                 {
//                     type: 'autogenerated',
//                     dirName: 'reference'
//                 }
//             ]
//         }
//     ],
// };

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
    // By default, Docusaurus generates a sidebar from the docs folder structure
    tutorialSidebar: [{type: 'autogenerated', dirName: '.'}],
  
    // But you can create a sidebar manually
    /*
    tutorialSidebar: [
      {
        type: 'category',
        label: 'Tutorial',
        items: ['hello'],
      },
    ],
     */
  };
  
  module.exports = sidebars;