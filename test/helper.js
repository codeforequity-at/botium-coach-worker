const fs = require('fs')
const path = require('path')

const readDefaultSet = () => [
  {
    intentName: 'greetings.bye',
    utterances: ['goodbye for now', 'bye bye take care', 'okay see you later', 'bye for now', 'i must go']
  },
  {
    intentName: 'greetings.hello',
    utterances: ['hello', 'hi', 'howdy']
  }
]

const readDataDir = (dataSetName) => {
  return fs.readdirSync(path.join(__dirname, '..', 'data', dataSetName)).map(filename => {
    const uf = path.join(__dirname, '..', 'data', dataSetName, filename)
    const uc = fs.readFileSync(uf).toString()
    const lines = uc.split('\n').map(l => l.trim())
    return {
      intentName: lines[0],
      utterances: lines.slice(1)
    }
  })
}

module.exports = {
  readDefaultSet,
  readDataDir
}
