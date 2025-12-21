// @ts-check

/** @type {import('@docusaurus/types').Plugin} */
module.exports = function chatbotPlugin(context, options) {
  return {
    name: 'docusaurus-plugin-chatbot',

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@chatbot': '../rag_frontend',
          },
        },
      };
    },

    getThemePath() {
      return './theme';
    },

    getClientModules() {
      return [require.resolve('./src/client-modules/chatbot')];
    },
  };
};