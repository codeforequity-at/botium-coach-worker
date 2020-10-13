const debug = require('debug')('botium-nlp-termanalytics')

const _adversarialExampleInference = async (utterance, classifactor) => {
  const adversarialResults = []

  const adversarialExamples = _generateAdversarialExamples(utterance)
  for (const adversarialExample of adversarialExamples) {
    const cls = await classifactor(adversarialExample.utterance)

    adversarialResults.push({
      utterance: adversarialExample.utterance,
      index: adversarialExample.index,
      results: cls
    })
  }
  return adversarialResults
}

const _generateAdversarialExamples = (utterance) => {
  const adversarialExamples = []
  const tokens = utterance.split(' ')
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

const _highlightScoring = (utterance, expectedIntentName, originalResult, adversarialResults) => {
  // console.log('_highlightScoring', expectedIntentName, originalResult, JSON.stringify(adversarialResults, null, 2))

  const _findIntentPosition = (results) => {
    const position = results.findIndex(r => r.intentName === expectedIntentName)
    if (position >= 0) return position
    else return results.length
  }

  const originalPosition = _findIntentPosition(originalResult)
  const originalScore = originalPosition < originalResult.length ? originalResult[originalPosition].score : 0.0

  // console.log('_highlightScoring original', originalPosition, originalScore)

  const tokens = utterance.split(' ')
  const highlight = tokens.map(token => ({ token, weight: 0.0 }))

  for (const adversarialResult of adversarialResults) {
    if (!adversarialResult.results || adversarialResult.results.length === 0) { continue }

    const adversarialPosition = _findIntentPosition(adversarialResult.results)
    const adversarialScore = adversarialPosition < adversarialResult.results.length ? adversarialResult.results[adversarialPosition].score : 0.0

    _scoring_function(highlight,
      originalPosition,
      adversarialPosition,
      originalScore,
      adversarialScore,
      adversarialResult.index)
  }
  debug(`_highlightScoring for utterance ${utterance}:`, highlight)
  return highlight
}

const _scoring_function = (highlight, originalPosition, adversarialPosition, originalScore, adversarialScore, index) => {
  // console.log('_scoring_function', index, originalPosition, adversarialPosition, originalScore, adversarialScore)
  const position_difference = (1 / (originalPosition + 1.0)) - (
    1 / (adversarialPosition + 1.0)
  )
  const confidence_difference = originalScore - adversarialScore
  const weighted_difference = (
    ((0.2 * confidence_difference) + (0.8 * position_difference))
  )
  // console.log('scoring', index, position_difference, confidence_difference, weighted_difference)
  highlight[index].weight += weighted_difference

  return highlight
}

const getHighlights = async (utterance, expectedIntentName, classifactor) => {
  const originalResult = await classifactor(utterance)
  const adversarialResults = await _adversarialExampleInference(utterance, classifactor)

  const highlights = _highlightScoring(utterance, expectedIntentName, originalResult, adversarialResults)
  return highlights
}

module.exports = {
  _adversarialExampleInference,
  _generateAdversarialExamples,
  getHighlights
}
