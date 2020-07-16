const assert = require('chai').assert
const fs = require('fs')
const nlpAnalytics = require('../src/nlpAnalytics')
const INPUT_POSTFIX = '.input.json'
const OUTPUT_POSTFIX = '.expectedOutput.json'
const TEST_DIR = '/dynamic/'

describe('dynamic', () => {
  // this.timeout(20000)
  fs.readdirSync(__dirname + TEST_DIR)
  .filter(file => file.endsWith(INPUT_POSTFIX))
  .map(file => file.substring(0, file.length - INPUT_POSTFIX.length))
  .forEach((suiteName) => {
    it(suiteName, async () => {
      const input = require('.' + TEST_DIR + suiteName + INPUT_POSTFIX)
      const expectedOutput = require('.' + TEST_DIR + suiteName + OUTPUT_POSTFIX)
      const output = await nlpAnalytics.process(input)
      assert.equal(Object.keys(output).sort(), Object.keys(expectedOutput).sort())

      for (const key of Object.keys(output)) {
        assert.deepEqual(output[key], expectedOutput[key])
      }
    })
  })
})
