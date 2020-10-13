const { Language, NlpManager } = require('node-nlp')
const _ = require('lodash')
const debug = require('debug')('botium-nlp-kfold')

const _flattenIntents = (intents) => intents.reduce((agg, intent) => [...agg, ...intent.utterances.map(utterance => ({ intentName: intent.intentName, utterance }))], [])

const _splitArray = (array = [], nPieces = 1) => {
  const splitArray = []
  let atArrPos = 0
  for (let i = 0; i < nPieces; i++) {
    const splitArrayLength = Math.ceil((array.length - atArrPos) / (nPieces - i))
    splitArray.push([])
    splitArray[i] = array.slice(atArrPos, splitArrayLength + atArrPos)
    atArrPos += splitArrayLength
  }
  return splitArray
}

const guessLanguage = async (text) => {
  const language = new Language()
  const guess = language.guess(text)
  return guess[0]
}

const guessLanguageForIntents = async (flattenedIntents) => {
  const text = flattenedIntents.map(f => f.utterance).join('\n')
  return guessLanguage(text)
}

const trainClassification = async (intents, lang) => {
  const flattened = _flattenIntents(intents)

  if (!lang) {
    lang = (await guessLanguageForIntents(flattened)).alpha2
    debug(`Identified language ${lang}`)
  }

  debug(`Training ${flattened.length} utterances ...`)
  const manager = new NlpManager({ languages: [lang], nlu: { log: false } })
  for (const f of flattened) {
    manager.addDocument(lang, f.utterance, f.intentName)
  }
  try {
    await manager.train()
    debug(`Training ${flattened.length} utterances ready`)
  } catch (err) {
    throw new Error(`Training ${flattened.length} utterances failed: ${err.message}`)
  }
  return async (text) => {
    try {
      const response = await manager.process(lang, text)
      return response.classifications.map(cl => ({
        intentName: cl.intent,
        score: cl.score
      }))
    } catch (err) {
      debug(`Classification for "${text}" failed: ${err.message}`)
      return []
    }
  }
}

const loocv = async (intents, lang) => {
  if (!lang) {
    const flattened = _flattenIntents(intents)
    lang = (await guessLanguageForIntents(flattened)).alpha2
    debug(`Identified language ${lang}`)
  }

  const allPromises = intents.reduce((agg, intent, iindex) => {
    const otherIntents = [...intents]
    otherIntents.splice(iindex, 1)
    const intentPromises = intent.utterances.map(async (utterance, uindex) => {
      const otherUtterances = [...intent.utterances]
      otherUtterances.splice(uindex, 1)

      const classifactor = await trainClassification([
        ...otherIntents,
        {
          intentName: intent.intentName,
          utterances: otherUtterances
        }
      ], lang)

      const response = await classifactor(utterance)
      const prediction = (response && response.length > 0 && response[0]) || null

      return {
        utterance,
        expectedIntent: intent.intentName,
        predictedIntent: prediction ? prediction.intentName : null,
        match: (prediction && prediction.intentName === intent.intentName) || false,
        score: (prediction && prediction.score) || 0.0
      }
    })
    return agg.concat(intentPromises)
  }, [])

  const results = await Promise.all(allPromises)
  const score = results.filter(r => r.match).length / results.length
  return {
    lang,
    score,
    results
  }
}

const makeFolds = (intents, { k, shuffle = false, onlyIntents = null } = {}) => {
  const result = []

  for (let i = 0; i < k; i++) {
    const chunk = intents.map(intent => {
      if (intent.utterances.length < k) {
        debug(`Intent ${intent.intentName} has too less utterances (${intent.utterances.length}), no folds created.`)
        return {
          intentName: intent.intentName,
          train: intent.utterances
        }
      }
      if (onlyIntents && onlyIntents.indexOf(intent.intentName) < 0) {
        debug(`Intent ${intent.intentName} is ignored, no folds created.`)
        return {
          intentName: intent.intentName,
          train: intent.utterances
        }
      }
      const chunks = _splitArray(shuffle ? _.shuffle(intent.utterances) : intent.utterances, k)
      return {
        intentName: intent.intentName,
        test: chunks[i],
        train: _.flatten(_.filter(chunks, (s, chunkIndex) => chunkIndex !== i))
      }
    })
    result.push(chunk)
  }
  return result
}

const makeSplit = async (intents, { ratio = 0.5, shuffle = false } = {}) => {
  const splittedUtterances = [
    [],
    []
  ]
  const round = ratio === 0.0 ? Math.floor : ratio === 1.0 ? Math.ceil : ratio > 0.5 ? Math.floor : Math.ceil

  for (const intent of intents) {
    const rnd = shuffle ? _.shuffle(intent.utterances) : intent.utterances

    const splitLength = round(intent.utterances.length * ratio)
    const utt0 = rnd.slice(0, splitLength)
    const utt1 = rnd.slice(splitLength)

    splittedUtterances[0].push({
      intentName: intent.intentName,
      utterances: utt0
    })
    splittedUtterances[1].push({
      intentName: intent.intentName,
      utterances: utt1
    })
  }
  return splittedUtterances
}

const runKFold = async (intents, { lang = null, k = 5, shuffle = false, onlyIntents = null } = {}) => {
  const folds = makeFolds(intents, { k, shuffle, onlyIntents })
  debug(`Created ${k} folds (shuffled: ${shuffle})`)
  if (!lang) {
    const flattened = _flattenIntents(intents)
    lang = (await guessLanguageForIntents(flattened)).alpha2
    debug(`Identified language ${lang}`)
  }

  const foldMatrices = []
  const predictionDetails = []

  for (let k = 0; k < folds.length; k++) {
    const foldIntents = folds[k]

    const trainingData = foldIntents.map(fi => ({
      intentName: fi.intentName,
      utterances: fi.train
    }))

    try {
      debug(`Starting training for fold ${k + 1}`)
      const classifactor = await trainClassification(trainingData, lang)

      debug(`Starting testing for fold ${k + 1}`)
      const testIntents = foldIntents.filter(fi => fi.test)

      const intentPromises = testIntents.map(async (foldIntent) => {
        foldIntent.predictions = {}

        for (const utterance of foldIntent.test) {
          const response = await classifactor(utterance)
          const prediction = (response && response.length > 0 && response[0]) || { intentName: 'None', score: 0.0 }

          foldIntent.predictions[prediction.intentName] = (foldIntent.predictions[prediction.intentName] || 0) + 1
          predictionDetails.push({
            fold: k,
            utterance,
            expectedIntent: foldIntent.intentName,
            predictedIntent: prediction.intentName,
            match: foldIntent.intentName === prediction.intentName,
            score: prediction.score
          })
        }
      })
      await Promise.all(intentPromises)

      const expectedIntents = {}
      for (const testIntent of testIntents) {
        expectedIntents[testIntent.intentName] = testIntent.predictions
      }
      const allIntentNames = foldIntents.map(fi => fi.intentName).concat(['None'])
      for (const intentName of allIntentNames) {
        expectedIntents[intentName] = expectedIntents[intentName] || { }
      }

      const matrix = []
      for (const testIntent of testIntents) {
        const totalPredicted = allIntentNames.reduce((agg, otherIntentName) => {
          return agg + (expectedIntents[otherIntentName][testIntent.intentName] || 0)
        }, 0)
        const totalExpected = allIntentNames.reduce((agg, otherIntentName) => {
          return agg + (testIntent.predictions[otherIntentName] || 0)
        }, 0)

        const score = { }

        if (totalPredicted === 0) {
          score.precision = 0
        } else {
          score.precision = (testIntent.predictions[testIntent.intentName] || 0) / totalPredicted
        }
        if (totalExpected === 0) {
          score.recall = 0
        } else {
          score.recall = (testIntent.predictions[testIntent.intentName] || 0) / totalExpected
        }
        if (score.precision === 0 && score.recall === 0) {
          score.F1 = 0
        } else {
          score.F1 = 2 * ((score.precision * score.recall) / (score.precision + score.recall))
        }

        matrix.push({
          intent: testIntent.intentName,
          predictions: testIntent.predictions,
          score
        })
      }

      const foldMatrix = {
        precision: matrix.reduce((sum, r) => sum + r.score.precision, 0) / matrix.length,
        recall: matrix.reduce((sum, r) => sum + r.score.recall, 0) / matrix.length,
        matrix
      }
      if (foldMatrix.precision === 0 && foldMatrix.recall === 0) {
        foldMatrix.F1 = 0
      } else {
        foldMatrix.F1 = 2 * ((foldMatrix.precision * foldMatrix.recall) / (foldMatrix.precision + foldMatrix.recall))
      }
      foldMatrices.push(foldMatrix)

      debug(`K-Fold Round ${k + 1}: Precision=${foldMatrix.precision.toFixed(4)} Recall=${foldMatrix.recall.toFixed(4)} F1-Score=${foldMatrix.F1.toFixed(4)}`)
    } catch (err) {
      debug(`K-Fold testing for fold ${k + 1} failed: ${err.message}`)
      throw new Error(`K-Fold testing for fold ${k + 1} failed: ${err.message}`)
    }
  }

  const avgPrecision = foldMatrices.reduce((sum, r) => sum + r.precision, 0) / foldMatrices.length
  const avgRecall = foldMatrices.reduce((sum, r) => sum + r.recall, 0) / foldMatrices.length
  let avgF1 = 0
  if (avgPrecision !== 0 || avgRecall !== 0) {
    avgF1 = 2 * ((avgPrecision * avgRecall) / (avgPrecision + avgRecall))
  }

  if (debug.enabled) {
    debug('############# Summary #############')
    for (let k = 0; k < foldMatrices.length; k++) {
      const foldMatrix = foldMatrices[k]
      debug(`K-Fold Round ${k + 1}: Precision=${foldMatrix.precision.toFixed(4)} Recall=${foldMatrix.recall.toFixed(4)} F1-Score=${foldMatrix.F1.toFixed(4)}`)
    }
    debug(`K-Fold Avg: Precision=${avgPrecision.toFixed(4)} Recall=${avgRecall.toFixed(4)} F1-Score=${avgF1.toFixed(4)}`)
  }

  return {
    lang,
    avgPrecision,
    avgRecall,
    avgF1,
    foldMatrices,
    predictionDetails
  }
}

const runValidation = async (trainIntents, testIntents, { lang = null } = {}) => {
  if (!lang) {
    const flattened = _flattenIntents(trainIntents)
    lang = (await guessLanguageForIntents(flattened)).alpha2
    debug(`Identified language ${lang}`)
  }

  const predictionDetails = []
  const validateMatrix = {
    lang,
    predictionDetails
  }

  try {
    debug('Starting training ...')
    const classifactor = await trainClassification(trainIntents, lang)

    debug('Starting testing ...')
    const intentPromises = testIntents.map(async (intent) => {
      intent.predictions = {}

      for (const utterance of intent.utterances) {
        const response = await classifactor(utterance)
        const prediction = (response && response.length > 0 && response[0]) || { intentName: 'None', score: 0.0 }

        intent.predictions[prediction.intentName] = (intent.predictions[prediction.intentName] || 0) + 1
        predictionDetails.push({
          utterance,
          expectedIntent: intent.intentName,
          predictedIntent: prediction.intentName,
          match: intent.intentName === prediction.intentName,
          score: prediction.score
        })
      }
    })
    await Promise.all(intentPromises)

    const expectedIntents = {}
    for (const testIntent of testIntents) {
      expectedIntents[testIntent.intentName] = testIntent.predictions
    }
    const allIntentNames = testIntents.map(fi => fi.intentName).concat(['None'])
    for (const intentName of allIntentNames) {
      expectedIntents[intentName] = expectedIntents[intentName] || { }
    }

    const matrix = []
    for (const testIntent of testIntents) {
      const totalPredicted = allIntentNames.reduce((agg, otherIntentName) => {
        return agg + (expectedIntents[otherIntentName][testIntent.intentName] || 0)
      }, 0)
      const totalExpected = allIntentNames.reduce((agg, otherIntentName) => {
        return agg + (testIntent.predictions[otherIntentName] || 0)
      }, 0)

      const score = { }

      if (totalPredicted === 0) {
        score.precision = 0
      } else {
        score.precision = (testIntent.predictions[testIntent.intentName] || 0) / totalPredicted
      }
      if (totalExpected === 0) {
        score.recall = 0
      } else {
        score.recall = (testIntent.predictions[testIntent.intentName] || 0) / totalExpected
      }
      if (score.precision === 0 && score.recall === 0) {
        score.F1 = 0
      } else {
        score.F1 = 2 * ((score.precision * score.recall) / (score.precision + score.recall))
      }

      matrix.push({
        intent: testIntent.intentName,
        predictions: testIntent.predictions,
        score
      })
    }
    validateMatrix.precision = matrix.reduce((sum, r) => sum + r.score.precision, 0) / matrix.length
    validateMatrix.recall = matrix.reduce((sum, r) => sum + r.score.recall, 0) / matrix.length
    validateMatrix.matrix = matrix
    if (validateMatrix.precision === 0 && validateMatrix.recall === 0) {
      validateMatrix.F1 = 0
    } else {
      validateMatrix.F1 = 2 * ((validateMatrix.precision * validateMatrix.recall) / (validateMatrix.precision + validateMatrix.recall))
    }

    if (debug.enabled) {
      debug('############# Summary #############')
      debug(`Validate: Precision=${validateMatrix.precision.toFixed(4)} Recall=${validateMatrix.recall.toFixed(4)} F1-Score=${validateMatrix.F1.toFixed(4)}`)
    }
    return validateMatrix
  } catch (err) {
    debug(`Validation failed: ${err.message}`)
    throw new Error(`Validation failed: ${err.message}`)
  }
}

module.exports = {
  guessLanguage,
  trainClassification,
  loocv,
  makeFolds,
  makeSplit,
  runKFold,
  runValidation
}
