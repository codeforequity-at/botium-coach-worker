const fs = require('fs')
const path = require('path')
const assert = require('chai').assert
const kFold = require('../src/kFold')

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

describe('kFold', function () {
  it('should recognize language', async () => {
    const lang1 = await kFold.guessLanguage('this is some english text')
    assert.equal(lang1.alpha2, 'en')
    const lang2 = await kFold.guessLanguage('cest la vie')
    assert.equal(lang2.alpha2, 'fr')
    const lang3 = await kFold.guessLanguage('das ist ein deutscher text')
    assert.equal(lang3.alpha2, 'de')
  })
  it('should train a basic model', async () => {
    const intents = readDefaultSet()
    const classificator = await kFold.trainClassification(intents)

    const r1 = await classificator('I should go now')
    assert.equal(r1[0].intentName, 'greetings.bye')
    const r2 = await classificator('hallo')
    assert.equal(r2[0].intentName, 'greetings.hello')
  })
  it('should run a basic loocv', async () => {
    const intents = readDefaultSet()
    const result = await kFold.loocv(intents)
    assert.isTrue(result.score >= 0 && result.score <= 1)
  })
  it('should run a large loocv', async () => {
    const intents = readDataDir('Insurance3')
    const result = await kFold.loocv(intents)
    assert.isTrue(result.score >= 0 && result.score <= 1)
  }).timeout(60000)
  it('should create folds', async () => {
    const intents = readDefaultSet()
    const folds = kFold.makeFolds(intents, { k: 5, shuffle: true })
    assert.equal(folds.length, 5)
    for (const fold of folds) {
      assert.equal(fold.length, 2)
      assert.equal(fold[0].intentName, 'greetings.bye')
      assert.equal(fold[0].test.length, 1)
      assert.equal(fold[0].train.length, 4)
      assert.equal(fold[1].intentName, 'greetings.hello')
      assert.isUndefined(fold[1].test)
      assert.equal(fold[1].train.length, 3)
    }
  })
  it('should create 100/0 split', async () => {
    const intents = readDefaultSet()
    const [train, test] = await kFold.makeSplit(intents, { ratio: 1.0 })
    assert.equal(train.length, 2)
    assert.equal(train[0].utterances.length, 5)
    assert.equal(train[1].utterances.length, 3)
    assert.equal(test.length, 2)
    assert.equal(test[0].utterances.length, 0)
    assert.equal(test[1].utterances.length, 0)
  })
  it('should create 0/100 split', async () => {
    const intents = readDefaultSet()
    const [train, test] = await kFold.makeSplit(intents, { ratio: 0.0 })
    assert.equal(train.length, 2)
    assert.equal(train[0].utterances.length, 0)
    assert.equal(train[1].utterances.length, 0)
    assert.equal(test.length, 2)
    assert.equal(test[0].utterances.length, 5)
    assert.equal(test[1].utterances.length, 3)
  })
  it('should create ~50/50 split', async () => {
    const intents = readDefaultSet()
    const [train, test] = await kFold.makeSplit(intents, { ratio: 0.5 })
    assert.equal(train.length, 2)
    assert.equal(train[0].utterances.length, 3)
    assert.equal(train[1].utterances.length, 2)
    assert.equal(test.length, 2)
    assert.equal(test[0].utterances.length, 2)
    assert.equal(test[1].utterances.length, 1)
  })
  it('should run a basic k-fold', async () => {
    const intents = readDefaultSet()
    const result = await kFold.runKFold(intents, { k: 2 })
    assert.isTrue(result.avgPrecision >= 0 && result.avgPrecision <= 1)
    assert.isTrue(result.avgRecall >= 0 && result.avgRecall <= 1)
    assert.isTrue(result.avgF1 >= 0 && result.avgF1 <= 1)
  })
  it('should run a large k-fold', async () => {
    const intents = readDataDir('InsuranceAll')
    const result = await kFold.runKFold(intents, { k: 5 })
    assert.isTrue(result.avgPrecision >= 0 && result.avgPrecision <= 1)
    assert.isTrue(result.avgRecall >= 0 && result.avgRecall <= 1)
    assert.isTrue(result.avgF1 >= 0 && result.avgF1 <= 1)
  })
  it('should run a basic validation', async () => {
    const intents = readDefaultSet()
    const [training, test] = await kFold.makeSplit(intents, { ratio: 0.5 })
    const result = await kFold.runValidation(training, test)
    assert.isTrue(result.precision >= 0 && result.precision <= 1)
    assert.isTrue(result.recall >= 0 && result.recall <= 1)
    assert.isTrue(result.F1 >= 0 && result.F1 <= 1)
  })
  it('should run a large validation', async () => {
    const intents = readDataDir('InsuranceAll')
    const [training, test] = await kFold.makeSplit(intents, { ratio: 0.8 })
    const result = await kFold.runValidation(training, test)
    assert.isTrue(result.precision >= 0 && result.precision <= 1)
    assert.isTrue(result.recall >= 0 && result.recall <= 1)
    assert.isTrue(result.F1 >= 0 && result.F1 <= 1)
  })
})
