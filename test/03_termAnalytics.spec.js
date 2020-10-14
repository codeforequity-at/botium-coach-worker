const assert = require('chai').assert
const kFold = require('../src/kFold')
const termAnalytics = require('../src/termAnalytics')
const { readDataDir, readDefaultSet } = require('./helper')

describe('termAnalysis', function () {
  it('should generate adversarial examples', async () => {
    const examples = termAnalytics._generateAdversarialExamples('goodbye for now', 'en')
    assert.equal(examples.length, 3)
  })
  it('should run adversarial examples', async () => {
    const intents = readDefaultSet()
    const classificator = await kFold.trainClassification(intents)

    const result = await termAnalytics._adversarialExampleInference('goodbye for now', classificator, 'en')
    assert.equal(result.length, 3)
  })
  it('should run a term analysis', async () => {
    const intents = readDataDir('InsuranceAll')
    const classificator = await kFold.trainClassification(intents)

    const result = await termAnalytics.getHighlights('i want to cancel my policy', 'INSURANCE.CANCEL_POLICY', classificator)
    assert.equal(result.length, 6)
  })
  it('should run multiple term analysis', async () => {
    const intents = readDataDir('InsuranceAll')
    const classificator = await kFold.trainClassification(intents)

    const cancelData = intents.find(i => i.intentName === 'INSURANCE.CANCEL_POLICY')

    const allResults = await termAnalytics.getAllHighlights(cancelData.utterances, cancelData.intentName, classificator, 'en')
    assert.isTrue(allResults.length > 0)
    assert.equal(allResults[0].token, 'kill')
  })
  it('should generate term analysis matrix', async () => {
    const intents = readDataDir('InsuranceAll')
    const classificator = await kFold.trainClassification(intents)

    const allResults = await termAnalytics.getHighlightsMatrix(intents, classificator, 'en')
    assert.isTrue(allResults.length > 0)
    assert.equal(allResults[0].token, 'approval')
  })
  it('should generate another term analysis matrix', async () => {
    const intents = readDataDir('Banking')
    const allResults = await termAnalytics.runHighlightsMatrix(intents)
    assert.equal(allResults.lang, 'en')
    assert.isTrue(allResults.matrix.length > 0)
    assert.equal(allResults.matrix[0].token, '2017')
  })
})
