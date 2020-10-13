const assert = require('chai').assert
const kFold = require('../src/kFold')
const termAnalytics = require('../src/termAnalytics')
const { readDataDir, readDefaultSet } = require('./helper')

describe('termAnalysis', function () {
  it('should generate adversarial examples', async () => {
    const examples = termAnalytics._generateAdversarialExamples('goodbye for now')
    assert.equal(examples.length, 3)
  })
  it('should run adversarial examples', async () => {
    const intents = readDefaultSet()
    const classificator = await kFold.trainClassification(intents)

    const result = await termAnalytics._adversarialExampleInference('goodbye for now', classificator)
    assert.equal(result.length, 3)
  })
  it('should run a term analysis', async () => {
    const intents = readDataDir('InsuranceAll')
    const classificator = await kFold.trainClassification(intents)

    const result = await termAnalytics.getHighlights('i want to cancel my policy', 'INSURANCE.CANCEL_POLICY', classificator)
    await termAnalytics.getHighlights('cancel my policy', 'INSURANCE.CANCEL_POLICY', classificator)
  })
})
