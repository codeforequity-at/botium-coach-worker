const nlpAnalytics = require('./src/nlpAnalytics')
const kFold = require('./src/kFold')
const termAnalytics = require('./src/termAnalytics')
const language = require('./src/language')

module.exports = {
  nlpAnalytics: {
    ...nlpAnalytics
  },
  kFold: {
    ...kFold
  },
  termAnalytics: {
    ...termAnalytics
  },
  language: {
    ...language
  }
}
