const assert = require('chai').assert
const fs = require('fs')
const path = require('path')
const _ = require('lodash')

const nlpAnalytics = require('../src/nlpAnalytics')

const INPUT_POSTFIX = '.input.json'
const OUTPUT_POSTFIX = '.expectedOutput.json'
const CONNECTOR_FEATURES_POSTFIX = '.connectorFeatures.json'
const TEST_DIR = 'dynamic'

const _assertForNanRecursive = (value, location) => {
  if (_.isNil(value)) {
    return
  }
  assert.isNotNaN(value, location)

  if (_.isArray(value)) {
    value.forEach((entry, index) => {
      _assertForNanRecursive(entry, `${location}[${index}]`)
    })
  }

  if (_.isObject(value)) {
    for (const [key, entry] of Object.entries(value)) {
      _assertForNanRecursive(entry, `${location}["${key}"]`)
    }
  }
}

describe('dynamic', () => {
  fs.readdirSync(path.join(__dirname, TEST_DIR))
    .filter(file => file.endsWith(INPUT_POSTFIX))
    .map(file => file.substring(0, file.length - INPUT_POSTFIX.length))
    .forEach((suiteName) => {
      it(suiteName, async () => {
        const input = require(path.join(__dirname, TEST_DIR, suiteName + INPUT_POSTFIX))
        const expectedOutput = require(path.join(__dirname, TEST_DIR, suiteName + OUTPUT_POSTFIX))
        let connectorFeatures = {}
        try {
          connectorFeatures = require(path.join(__dirname, TEST_DIR, suiteName + CONNECTOR_FEATURES_POSTFIX))
        } catch (err) {}
        const output = await nlpAnalytics.process({ testCaseResults: input, connectorFeatures })

        // console.log(JSON.stringify(output))
        _assertForNanRecursive(output, '')

        assert.deepEqual(Object.keys(output).sort(), Object.keys(expectedOutput).sort())

        for (const key of Object.keys(output)) {
          // JSON.parse + JSON.stringify: we are not able to store undefined in a json file.
          // So we are removing from output too
          // fs.writeFileSync('delme\\' + key + '.json', JSON.stringify(output[key]), 'utf8')
          assert.deepEqual(JSON.parse(JSON.stringify(output[key])), expectedOutput[key], `The field ${key} does not match`)
        }
      })
    })
})
