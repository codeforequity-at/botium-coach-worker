const _ = require('lodash')
const debug = require('debug')('botium-nlp-termanalytics')

const { guessLanguageForIntents, removeStopwords, tokenize } = require('./language')
const { trainClassification } = require('./kFold')

const _adversarialExampleInference = async (utterance, classificator, lang) => {
  const adversarialResults = []

  const adversarialExamples = _generateAdversarialExamples(utterance, lang)
  for (const adversarialExample of adversarialExamples) {
    const cls = await classificator(adversarialExample.utterance)

    adversarialResults.push({
      utterance: adversarialExample.utterance,
      index: adversarialExample.index,
      results: cls
    })
  }
  return adversarialResults
}

const _generateAdversarialExamples = (utterance, lang) => {
  const adversarialExamples = []
  const tokens = tokenize(utterance, lang)
  tokens.forEach((token, index) => {
    const newExample = [...tokens]
    newExample.splice(index, 1)
    adversarialExamples.push({
      utterance: newExample.join(' '),
      index
    })
  })
  return adversarialExamples
}

const _highlightScoring = (utterance, expectedIntentName, originalResult, adversarialResults, lang) => {
  // console.log('_highlightScoring', expectedIntentName, originalResult, JSON.stringify(adversarialResults, null, 2))

  const _findIntentPosition = (results) => {
    const position = results.findIndex(r => r.intentName === expectedIntentName)
    if (position >= 0) return position
    else return results.length
  }

  const originalPosition = _findIntentPosition(originalResult)
  const originalScore = originalPosition < originalResult.length ? originalResult[originalPosition].score : 0.0

  // console.log('_highlightScoring original', originalPosition, originalScore)

  const tokens = tokenize(utterance, lang)
  const highlight = tokens.map(token => ({ token, weight: 0.0 }))

  for (const adversarialResult of adversarialResults) {
    if (!adversarialResult.results || adversarialResult.results.length === 0) { continue }

    const adversarialPosition = _findIntentPosition(adversarialResult.results)
    const adversarialScore = adversarialPosition < adversarialResult.results.length ? adversarialResult.results[adversarialPosition].score : 0.0

    _addHighlightScoring(highlight,
      originalPosition,
      adversarialPosition,
      originalScore,
      adversarialScore,
      adversarialResult.index)
  }
  // debug(`_highlightScoring for utterance ${utterance}:`, highlight)
  return highlight
}

const _addHighlightScoring = (highlight, originalPosition, adversarialPosition, originalScore, adversarialScore, adversarialIndex) => {
  const positionDifference = (1.0 / (originalPosition + 1.0)) - (
    1.0 / (adversarialPosition + 1.0)
  )
  const confidenceDifference = originalScore - adversarialScore
  const weightedDifference = (
    ((0.2 * confidenceDifference) + (0.8 * positionDifference))
  )
  // debug('_addHighlightScoring', adversarialIndex, originalPosition, adversarialPosition, originalScore, adversarialScore)
  // debug('_addHighlightScoring weights', adversarialIndex, positionDifference, confidenceDifference, weightedDifference)
  highlight[adversarialIndex].weight += weightedDifference
}

const getHighlights = async (utterance, expectedIntentName, classificator, lang) => {
  const originalResult = await classificator(utterance)
  const adversarialResults = await _adversarialExampleInference(utterance, classificator, lang)

  const highlights = _highlightScoring(utterance, expectedIntentName, originalResult, adversarialResults, lang)
  return highlights
}

const getAllHighlights = async (utterances, expectedIntentName, classificator, lang) => {
  const allTokens = {}
  for (const utterance of utterances) {
    const originalResult = await classificator(utterance)
    const adversarialResults = await _adversarialExampleInference(utterance, classificator, lang)
    const highlights = _highlightScoring(utterance, expectedIntentName, originalResult, adversarialResults, lang)
    highlights.forEach(h => {
      if (allTokens[h.token]) allTokens[h.token] += h.weight
      else allTokens[h.token] = h.weight
    })
  }
  const allTokensList = removeStopwords(Object.keys(allTokens), lang).map(token => ({ token, weight: allTokens[token] }))
  const allTokensListSorted = _.orderBy(allTokensList.filter(t => t.weight > 0), ['weight'], ['desc'])
  return allTokensListSorted
}

const getHighlightsMatrix = async (intents, classificator, lang) => {
  const allTokensByIntent = {}
  for (const intent of intents) {
    const intentTokens = await getAllHighlights(intent.utterances, intent.intentName, classificator, lang)

    for (const intentToken of intentTokens) {
      if (intentToken.weight > 0.01) {
        if (allTokensByIntent[intentToken.token]) {
          allTokensByIntent[intentToken.token][intent.intentName] = intentToken.weight
        } else {
          allTokensByIntent[intentToken.token] = {
            [intent.intentName]: intentToken.weight
          }
        }
      }
    }
  }
  const allTokensList = removeStopwords(Object.keys(allTokensByIntent), lang).map(token => ({ token, weights: allTokensByIntent[token] }))
  const allTokensListSorted = _.orderBy(allTokensList, ['token'], ['asc'])
  return allTokensListSorted
}

const runHighlightsMatrix = async (intents, { lang = null } = {}) => {
  if (!lang) {
    lang = await guessLanguageForIntents(intents)
    debug(`Identified language ${lang}`)
  }

  const classificator = await trainClassification(intents, { lang })
  return {
    lang,
    matrix: await getHighlightsMatrix(intents, classificator, lang)
  }
}

module.exports = {
  _adversarialExampleInference,
  _generateAdversarialExamples,
  getHighlights,
  getAllHighlights,
  getHighlightsMatrix,
  runHighlightsMatrix
}
