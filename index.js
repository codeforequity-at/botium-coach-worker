const nlpAnalytics = require('./src/nlpAnalytics')
const kFold = require('./src/kFold')

module.exports = {
  nlpAnalytics: {
    ...nlpAnalytics
  },
  kFold: {
    ...kFold
  }
}
