const _ = require('lodash')
const debug = require('debug')('botium-nlp-analytics')

const { ENTITY_WITHOUT_NAME } = '<no name>'

const _calculateConfidenceDistibution = (context) => {
  const {
    result: {
      perUtterance
    }
  } = context

  const result = {}
  for (const u of perUtterance) {
    if (!u.intent.actual) {
      continue
    }
    if (!_.isNumber(u.intent.confidenceReal) || !_.isNumber(u.intent.confidenceCorrected)) {
      continue
    }

    const updateMapForField = (idToRecord, intent, confidence, fieldname, step) => {
      const calculateStepIndex = (confidence, step) => {
        const maxStepCount = 100 / step
        const stepIndexRaw = Math.trunc(confidence * maxStepCount)
        // first group: 0 <= x < 0.1. And so for every group. Except last:
        // 0.9 <= x <= 1
        return stepIndexRaw === maxStepCount ? (maxStepCount - 1) : stepIndexRaw
      }
      const stepIndex = calculateStepIndex(confidence, step)
      const mapId = JSON.stringify({ intent, step, stepIndex })

      if (!idToRecord[mapId]) {
        idToRecord[mapId] = {
          count: 0,
          countCorrected: 0,
          countPredictedCorrect: 0,
          countPredictedIncorrect: 0,
          intent,
          step,
          stepIndex
        }
      }
      idToRecord[mapId][fieldname]++
    }

    const updateMap = (utteranceStruct, step) => {
      if (_.isNumber(utteranceStruct.intent.confidenceReal)) {
        if (!utteranceStruct.intent.incomprehension) {
          updateMapForField(result, utteranceStruct.intent.actual, utteranceStruct.intent.confidenceReal, 'count', step)
          const intentCorrected = utteranceStruct.intent.expected || utteranceStruct.intent.actual
          updateMapForField(result, intentCorrected, utteranceStruct.intent.confidenceCorrected, 'countCorrected', step)
        }
        if (utteranceStruct.intent.expected ? (utteranceStruct.intent.actual === utteranceStruct.intent.expected) : !utteranceStruct.intent.incomprehension) {
          // intent has not many sense in confidence threshold chart. utteranceStruct.intent.actual could be even null, or expected
          updateMapForField(result, utteranceStruct.intent.actual, utteranceStruct.intent.confidenceReal, 'countPredictedCorrect', step)
        } else {
          updateMapForField(result, utteranceStruct.intent.actual, utteranceStruct.intent.confidenceReal, 'countPredictedIncorrect', step)
        }
      }
    }
    updateMap(u, 5)
    updateMap(u, 10)
  }

  context.result.confidenceDistribution = Object.values(result)
}

const _calculateUnexpectedIntentByAlternativeList = (context) => {
  const {
    result: {
      perUtterance
    }
  } = context
  const keyToConfidences = perUtterance
    .filter((u) => {
      return u.intent.actual && u.intent.actuals.length
    }).reduce((keyToConfidences, u) => {
      if (u.intent.confidenceReal) {
        for (const alter of u.intent.actuals) {
          if (alter.name !== u.intent.actual && alter.confidence) {
            const key = JSON.stringify([alter.name, u.intent.actual].sort())
            if (!keyToConfidences[key]) {
              keyToConfidences[key] = []
            }
            keyToConfidences[key].push({ confidence: (alter.confidence / u.intent.confidenceReal), utteranceStruct: u })
          }
        }
      }

      return keyToConfidences
    },
    []
    )

  let maxCount = 0
  for (const confidences of Object.values(keyToConfidences)) {
    maxCount = Math.max(maxCount, confidences.length)
  }

  const result = []
  for (const key of Object.keys(keyToConfidences)) {
    const [intent1, intent2] = JSON.parse(key)
    const confidences = keyToConfidences[key]
    let sum = 0
    for (const confidenceStruct of confidences) {
      sum += confidenceStruct.confidence || 0
    }
    const sorted = _.sortBy(confidences, c => -c.confidence)
    result.push({
      intent1,
      intent2,
      intent1And2: key,
      count: sorted.length,
      avg: sum / maxCount,
      worstConfidence: sorted[0].confidence,
      worstUtterance: sorted[0].utteranceStruct.utterance
    })
  }
  context.result.unexpectedIntentByAlternativeList = result
}

const _countUnexpectedIntents = (context) => {
  const {
    result: {
      perUtterance
    }
  } = context

  const intentsToStruct = perUtterance
    .filter((u) => {
      return u.intent.expected && u.intent.actual && (u.intent.expected !== u.intent.actual)
    }).filter((u) => {
      return !u.intent.incomprehension
    }).map(u => {
      return [u.intent.expected, u.intent.actual].sort()
    }).reduce((accumulator, currentValue) => {
      const key = JSON.stringify(currentValue)
      accumulator[key] = {
        intent1: currentValue[0],
        intent2: currentValue[1],
        intent1And2: JSON.stringify(currentValue),
        count: accumulator[key] ? accumulator[key].count + 1 : 1
      }
      return accumulator
    }, {})
  context.result.unexpectedIntent = Object.values(intentsToStruct)
}

const _calculateConfidenceThreshold = (context) => {
  const {
    result: {
      perUtterance
    }
  } = context

  const result = []
  for (let i = 0; i <= 100; i++) {
    result.push({
      confidenceThreshold: i / 100,
      truePositive: 0,
      trueNegative: 0,
      falsePositive: 0,
      falseNegative: 0
    })
  }
  for (const utt of perUtterance) {
    const confidence = utt.intent.confidenceReal
    if (_.isNumber(confidence)) {
      const real = utt.intent.expected ? (utt.intent.actual === utt.intent.expected) : !utt.intent.incomprehension
      for (let i = 0; i <= 100; i++) {
        const guess = confidence >= i / 100
        if (guess) {
          if (real) {
            result[i].truePositive++
          } else {
            result[i].falsePositive++
          }
        } else {
          if (real) {
            result[i].falseNegative++
          } else {
            result[i].trueNegative++
          }
        }
      }
    }
  }
  context.result.confidenceThreshold = Object.values(result)
}

const _addSafe = (n1, n2) => {
  if (_.isNil(n1) && _.isNil(n2)) {
    return null
  }
  if (_.isNil(n1)) {
    return n2
  }
  if (_.isNil(n2)) {
    return n1
  }
  return n1 + n2
}

const _calculateAggregatedStats = (context) => {
  const {
    result: {
      perExpectedEntity,
      perActualEntity,
      perExpectedIntent,
      perCorrectedIntent,
      perActualIntent,
      perUtterance
    }
  } = context
  for (const entry of Object.values(perExpectedIntent)) {
    entry.avg = (_.isNil(entry.sum) || entry.count === 0) ? null : entry.sum / entry.count
  }

  for (const entry of Object.values(perExpectedEntity)) {
    entry.avg = (_.isNil(entry.sum) || entry.count === 0) ? null : (entry.sum / entry.count)
  }

  for (const entry of Object.values(perActualIntent)) {
    entry.avg = (_.isNil(entry.sum) || entry.count === 0) ? null : entry.sum / entry.count
  }

  for (const entry of Object.values(perActualEntity)) {
    entry.avg = (_.isNil(entry.sum) || entry.count === 0) ? null : (entry.sum / entry.count)
  }

  for (const entry of Object.values(perCorrectedIntent)) {
    entry.avg = (_.isNil(entry.sum) || entry.count === 0) ? null : entry.sum / entry.count
  }

  context.result.overallStat = Object.assign(
    context.result.overallStat,
    {
      count: perUtterance.length,
      avg: (!_.isNumber(context.result.overallStat.sum) || context.result.overallStat.stepsWithActualIntentAndConfidence === 0) ? null : context.result.overallStat.sum / context.result.overallStat.stepsWithActualIntentAndConfidence,
      avgReal: (!_.isNumber(context.result.overallStat.sumReal) || context.result.overallStat.stepsWithActualIntentAndConfidence === 0) ? null : context.result.overallStat.sumReal / context.result.overallStat.stepsWithActualIntentAndConfidence,
      missedIntentsAvg: (!_.isNumber(context.result.overallStat.missedIntentsSum) || context.result.overallStat.missedIntents === 0) ? null : context.result.overallStat.missedIntentsSum / context.result.overallStat.missedIntents,
      confidenceDifferenceSquaredSum: 0,
      expectedIntentsSupported: Object.keys(perExpectedIntent).length > 0,
      expectedEntitiesSupported: Object.keys(perExpectedEntity).length > 0
    }
  )

  if (_.isNumber(context.result.overallStat.avgReal) && _.isNumber(context.result.overallStat.missedIntentsAvg)) {
    if (context.result.overallStat.avgReal <= context.result.overallStat.missedIntentsAvg) {
      context.result.overallStat.confidenceScoreReliability = 0
    } else {
      // real missed diff confidenceScoreReliability
      // 0.5  0.6    0.0  0.00
      // 0.6  0.55   0.05 0.22
      // 0.6  0.5    0.1  0.31
      // 0.6  0.45   0.25 0.50
      // 0.6  0.1    0.5  0.70
      // 1.0  0.0    1.0  1.00

      context.result.overallStat.confidenceScoreReliability = Math.sqrt(context.result.overallStat.missedIntentsAvg - context.result.overallStat.avgReal)
    }
  } else {
    context.result.overallStat.confidenceScoreReliability = null
  }
  context.result.overallStat.confidenceScoreReliability =

    perUtterance.forEach((entry) => {
      if (context.result.overallStat.intentConfidenceSupported) {
        if (entry.intent.expected) {
          const intentStatExpected = perExpectedIntent[entry.intent.expected]
          const confidenceDifferenceExpected = entry.intent.confidenceCorrected - intentStatExpected.avg
          entry.intent.confidenceDifferenceExpected = confidenceDifferenceExpected
          intentStatExpected.confidenceDifferenceSquaredSum += _defaultIfNotNumber(confidenceDifferenceExpected * confidenceDifferenceExpected)
        }

        if (entry.intent.actual) {
          const intentStatCorrected = perCorrectedIntent[entry.intent.expected || entry.intent.actual]
          const confidenceDifferenceCorrected = entry.intent.confidenceCorrected - intentStatCorrected.avg
          entry.intent.confidenceDifferenceCorrected = confidenceDifferenceCorrected
          intentStatCorrected.confidenceDifferenceSquaredSum += _defaultIfNotNumber(confidenceDifferenceCorrected * confidenceDifferenceCorrected)

          const intentStatActual = perActualIntent[entry.intent.actual]
          const confidenceDifferenceActual = entry.intent.confidenceReal - intentStatActual.avg
          entry.intent.confidenceDifferenceActual = confidenceDifferenceActual
          intentStatActual.confidenceDifferenceSquaredSum += _defaultIfNotNumber(confidenceDifferenceActual * confidenceDifferenceActual)
          context.result.overallStat.confidenceDifferenceSquaredSum += _defaultIfNotNumber(confidenceDifferenceActual * confidenceDifferenceActual)
        }
      }
      if (context.result.overallStat.entityConfidenceSupported) {
        for (const entity of entry.entity.expected) {
          const entityStat = perExpectedEntity[entity.name]
          const confidenceDifferenceExpected = _.isNil(entityStat.avg) ? null : entity.confidenceCorrected - entityStat.avg
          entityStat.confidenceDifferenceSquaredSum += _defaultIfNotNumber(confidenceDifferenceExpected * confidenceDifferenceExpected)
        }
        for (const entity of entry.entity.actual) {
          const entityStat = perActualEntity[entity.name]
          const confidenceDifferenceExpected = _.isNil(entityStat.avg) ? null : entity.confidence - entityStat.avg
          entityStat.confidenceDifferenceSquaredSum += _defaultIfNotNumber(confidenceDifferenceExpected * confidenceDifferenceExpected)
        }
      }
    })
}

const _calculateDeviation = (context) => {
  const {
    result: {
      overallStat,
      perExpectedEntity,
      perActualEntity,
      perExpectedIntent,
      perCorrectedIntent,
      perActualIntent
    }
  } = context
  if (overallStat.intentConfidenceSupported) {
    Object.values(perExpectedIntent).forEach((entry) => {
      entry.deviation = _.isNil(entry.confidenceDifferenceSquaredSum) ? null : Math.sqrt(entry.confidenceDifferenceSquaredSum / entry.count)
    })
    Object.values(perActualIntent).forEach((entry) => {
      entry.deviation = _.isNil(entry.confidenceDifferenceSquaredSum) ? null : Math.sqrt(entry.confidenceDifferenceSquaredSum / entry.count)
    })
    Object.values(perCorrectedIntent).forEach((entry) => {
      entry.deviation = _.isNil(entry.confidenceDifferenceSquaredSum) ? null : Math.sqrt(entry.confidenceDifferenceSquaredSum / entry.count)
    })
  }

  if (overallStat.entityConfidenceSupported) {
    Object.values(perExpectedEntity).forEach((entry) => {
      entry.deviation = _.isNil(entry.confidenceDifferenceSquaredSum) ? null : Math.sqrt(entry.confidenceDifferenceSquaredSum / entry.count)
    })
    Object.values(perActualEntity).forEach((entry) => {
      entry.deviation = _.isNil(entry.confidenceDifferenceSquaredSum) ? null : Math.sqrt(entry.confidenceDifferenceSquaredSum / entry.count)
    })
  }
  // overallStat.deviation = Math.sqrt(overallStat.confidenceDifferenceSquaredSum / overallStat.count)
}

const _calculateConfusionMatrix = (context) => {
  const {
    result: {
      overallStat,
      perExpectedIntent,
      perCorrectedIntent,
      perActualIntent,
      perUtterance
    }
  } = context
  const expectedIntents = {}
  const listIntentNames = []
  let processableResponses = 0
  for (const utt of perUtterance) {
    if (utt.intent.actual) listIntentNames.push(utt.intent.actual)
    if (utt.intent.expected) listIntentNames.push(utt.intent.expected)

    if (utt.intent.actual && utt.intent.expected) {
      expectedIntents[utt.intent.expected] = expectedIntents[utt.intent.expected] || {}
      expectedIntents[utt.intent.expected][utt.intent.actual] = (expectedIntents[utt.intent.expected][utt.intent.actual] || 0) + 1
      processableResponses++
    }
  }
  const allIntentNames = _.uniq(listIntentNames)
  for (const intentName of allIntentNames) {
    expectedIntents[intentName] = expectedIntents[intentName] || {}
  }

  const matrix = []
  let totalTruePositive = 0
  for (const intentName of allIntentNames) {
    const intentRow = expectedIntents[intentName]

    const score = {}
    if (Object.keys(intentRow).length) {
      const totalPredicted = allIntentNames.reduce((agg, otherIntentName) => {
        return agg + (expectedIntents[otherIntentName][intentName] || 0)
      }, 0)
      const totalExpected = allIntentNames.reduce((agg, otherIntentName) => {
        return agg + (intentRow[otherIntentName] || 0)
      }, 0)
      const truePositive = intentRow[intentName] || 0
      totalTruePositive += truePositive
      const trueNegative = processableResponses - (totalExpected + totalPredicted - truePositive)

      score.accuracy = processableResponses ? ((truePositive + trueNegative) / processableResponses) : 1
      if (totalPredicted === 0) {
        score.precision = 1
      } else {
        score.precision = truePositive / totalPredicted
      }
      if (totalExpected === 0) {
        score.recall = 1
      } else {
        score.recall = truePositive / totalExpected
      }
      score.F1 = (score.precision + score.recall) === 0 ? null : 2 * ((score.precision * score.recall) / (score.precision + score.recall))
    } else {
      score.accuracy = null
      score.precision = null
      score.recall = null
      score.F1 = null
    }

    matrix.push({
      intent: intentName,
      row: intentRow,
      score
    })

    if (perActualIntent[intentName]) {
      Object.assign(perActualIntent[intentName], score)
    }
    if (perExpectedIntent[intentName]) {
      Object.assign(perExpectedIntent[intentName], score)
    }
    if (perCorrectedIntent[intentName]) {
      Object.assign(perCorrectedIntent[intentName], score)
    }
  }

  const acceptedIntents = Object.keys(expectedIntents).length

  if (processableResponses) {
    overallStat.accuracy = totalTruePositive / processableResponses
    overallStat.precision = matrix.reduce((sum, r) => sum + (r.score.precision || 0), 0) / acceptedIntents
    overallStat.recall = matrix.reduce((sum, r) => sum + (r.score.recall || 0), 0) / acceptedIntents
    overallStat.F1 = 2 * ((overallStat.precision * overallStat.recall) / (overallStat.precision + overallStat.recall))
    overallStat.confusionMatrixStrength = processableResponses / perUtterance.length
  } else {
    overallStat.accuracy = null
    overallStat.precision = null
    overallStat.recall = null
    overallStat.F1 = null
    overallStat.confusionMatrixStrength = null
  }

  context.result.confusionMatrix = matrix
}

const _defaultIfNotNumber = (maybeNumber, def = null) => {
  if (_.isNumber(maybeNumber)) {
    return maybeNumber
  }

  return def
}

const _isIncomprehensionByIncompUtterance = (entry, stepIndex) => {
  if (!entry.testSet.expandUtterancesIncomprehension) {
    return false
  }

  const botUtterance = JSON.parse(entry.steps[stepIndex].expected).messageText || ''
  if (botUtterance !== entry.testSet.expandUtterancesIncomprehension) {
    return false
  }

  if (!entry.errDetails) {
    return false
  }
  const errDetails = JSON.parse(entry.errDetails)
  if (!errDetails) {
    return false
  }
  const errDetailsIncomprehension = errDetails.find(e => {
    return (e.source === 'TextMatchAsserter') && e.cause && e.cause.not
  })

  return !!errDetailsIncomprehension
}

const _processTestCaseResult = (entry, stepIndex, context) => {
  const {
    utteranceListProcessor,
    result: {
      ignoredTestScripts,
      overallStat,
      ignoredTestScriptSteps,
      perExpectedEntity,
      perActualEntity,
      perExpectedIntent,
      perCorrectedIntent,
      perActualIntent,
      perUtterance
    }
  } = context

  utteranceListProcessor.setTestCaseResult(entry)
  /*
  checks now just incomprehension utterance asserters
   */
  if (!entry.steps || entry.steps.length < 1) {
    ignoredTestScripts.push({ resultId: entry.id, testcaseName: entry.testcaseName, err: 'No steps' })
    overallStat.testScriptsIgnored++
  } else {
    let processableStepFound
    let processedStepFound
    for (let stepIndex = 1; stepIndex < entry.steps.length; stepIndex++) {
      const step = entry.steps[stepIndex]
      // Extract raw data from step
      if (step.sender === 'bot') {
        processableStepFound = true
        let criticalErrors = []
        if (step.errDetails) {
          // If some other asserter causes the error, then it is not important for us, we can evaluate the result
          try {
            criticalErrors = JSON.parse(step.errDetails).filter(entry => entry.type !== 'asserter').map(entry => JSON.stringify(entry))
          } catch (err) {
            criticalErrors.push(step.errDetails)
          }
        }

        overallStat.steps++
        if (criticalErrors.length) {
          ignoredTestScriptSteps.push({ stepId: step.id || step.step, stepIndex, err: criticalErrors })
          overallStat.stepsIgnored++
        } else {
          // extract basic info from step
          let utterance = ''
          if (stepIndex && entry.steps[stepIndex - 1].sender === 'me') {
            try {
              utterance = JSON.parse(entry.steps[stepIndex - 1].actual).messageText || ''
            } catch (err) {
            }
          } else {
            debug(`Cant find "me" for "bot" ${JSON.stringify(step)} in step ${JSON.stringify(entry.steps[stepIndex - 1])}`)
          }

          const actualIncomprehensionIncompUtterance = _isIncomprehensionByIncompUtterance(entry, stepIndex)

          if (step.actualIntents && step.actualIntents.length > 0 && !step.actualIntents[0].name) {
            step.actualIntents[0].name = 'None'
            step.actualIntents[0].incomprehension = true
            debug(`Intent without name in utterance "${utterance}" step "${step.id || step.step}"!`)
          }

          const actualIntent = step.actualIntents && step.actualIntents.length > 0 ? step.actualIntents[0].name : 'None'
          const actualIncomprehensionIntent = step.actualIntents && step.actualIntents.length > 0 ? step.actualIntents[0].incomprehension : false

          const actualIntents = step.actualIntents || []
          for (const intent of actualIntents) {
            const intentResolutionSupported = !_.isNull(intent.name)
            // we set this flag just to true, we can be never sure its not supported.
            if (intentResolutionSupported) {
              overallStat.intentResolutionSupported = true
            }

            const intentConfidenceSupported = !_.isNull(intent.confidence)
            // we set this flag just to true, we can be never sure its not supported.
            if (intentConfidenceSupported) {
              overallStat.intentConfidenceSupported = true
            }
          }

          const confidenceReal = step.actualIntents && step.actualIntents.length > 0 ? step.actualIntents[0].confidence : 0
          const expectedIntent = step.expectedIntent
          const correctedIntent = expectedIntent || actualIntent

          let confidenceCorrected
          let correctedIncomprehensionIntent
          if (!expectedIntent || (actualIntent === expectedIntent)) {
            confidenceCorrected = confidenceReal
            correctedIncomprehensionIntent = actualIncomprehensionIntent
          } else {
            confidenceCorrected = 0
            // we cant say for expected intent that it is incomprehension
            correctedIncomprehensionIntent = null
          }

          let expectedEntityNames = step.expectedEntities
          let acceptMoreEntities
          if (expectedEntityNames.length && ['...', '..'].includes(expectedEntityNames[expectedEntityNames.length - 1])) {
            acceptMoreEntities = true
            expectedEntityNames = expectedEntityNames.slice(0, expectedEntityNames.length - 1)
          }
          let expectedEntityValues = step.expectedEntityValues
          if (expectedEntityValues.length && ['...', '..'].includes(expectedEntityValues[expectedEntityValues.length - 1])) {
            expectedEntityValues = expectedEntityValues.slice(0, expectedEntityValues.length - 1)
          }

          const createActualEntities = () => {
            const nameToCount = {}
            const result = []
            for (const actualEntity of step.actualEntities) {
              if (!actualEntity.name) {
                debug(`Entity without name in utterance "${utterance}" step "${step.id || step.step}"!`)
              }
              const name = actualEntity.name || ENTITY_WITHOUT_NAME
              const count = (nameToCount[name] || 0)
              nameToCount[name] = count
              result.push(Object.assign({ entityKey: `${actualEntity.name} - ${count}` }, Object.assign(actualEntity, { name })))
            }
            return result
          }
          const actualEntities = createActualEntities()
          const actualEntityKeys = actualEntities.map(e => e.entityKey)
          const actualEntityNames = actualEntities.map(entity => entity.name)
          const createExpectedEntityNameToKey = () => {
            const result = []
            const nameToCount = {}
            for (const e of expectedEntityNames) {
              const count = (nameToCount[e] || 0)
              nameToCount[e] = count
              result.push(`${e} - ${count}`)
            }
            return result
          }
          const expectedEntityKeys = createExpectedEntityNameToKey()

          const justExpectedEntityKeys = expectedEntityKeys.filter((name) => !actualEntityKeys.includes(name))
          const expectedAndActualEntityKeys = expectedEntityKeys.filter((name) => actualEntityKeys.includes(name))
          const justActualEntityKeys = actualEntityKeys.filter((name) => !expectedEntityKeys.includes(name))

          // overall stat
          if (actualIntent) {
            overallStat.stepsWithActualIntent++
            if (_.isNumber(confidenceReal)) {
              overallStat.stepsWithActualIntentAndConfidence++
            }
          }
          if (actualIncomprehensionIntent || actualIncomprehensionIncompUtterance) {
            overallStat.incomprehensionIntents++
          }

          if (expectedIntent) {
            overallStat.stepsWithIntent++
            if (actualIntent !== expectedIntent) {
              overallStat.missedIntents++
              overallStat.missedIntentsSum = _addSafe(overallStat.missedIntentsSum, confidenceReal)
            } else {
              overallStat.matchedIntents++
            }
          }
          if (expectedEntityNames.length) {
            overallStat.stepsWithEntity++
          }
          if (expectedEntityValues.length) {
            overallStat.stepsWithEntityValue++
          }
          if (!processedStepFound) {
            overallStat.testScriptsProcessed++
            processedStepFound = true
          }

          const intentListSupported = actualIntents && actualIntents.length > 1
          // we set this flag just to true, we can be never sure its not supported.
          if (intentListSupported) {
            overallStat.intentListSupported = true
          }
          overallStat.justExpectedEntities += justExpectedEntityKeys.length
          overallStat.expectedAndActualEntities += acceptMoreEntities ? actualEntityNames.length : expectedAndActualEntityKeys.length
          overallStat.justActualEntitiesChecked += (acceptMoreEntities || !expectedEntityNames.length) ? 0 : justActualEntityKeys.length
          overallStat.justActualEntitiesNotChecked += (acceptMoreEntities || expectedEntityNames.length) ? 0 : justActualEntityKeys.length
          if (actualEntities && actualEntities.length) {
            for (const e of actualEntities) {
              if (!acceptMoreEntities && justActualEntityKeys.includes(e.entityKey)) {
                if (expectedEntityNames.length) {
                  e.justActualAndChecked = true
                } else {
                  e.justActualAndNotChecked = true
                }
              }

              const entityResolutionSupported = !_.isNil(e.name)
              // we set this flag just to true, we can be never sure its not supported.
              if (entityResolutionSupported) {
                overallStat.entityResolutionSupported = true
              }

              const entityConfidenceSupported = _.isNumber(e.confidence)
              // we set this flag just to true, we can be never sure its not supported.
              if (entityConfidenceSupported) {
                overallStat.entityConfidenceSupported = true
              }
            }
          }

          // perExpectedEntity
          const actualEntitiesTemp = [...actualEntities]
          // at matching actual and expected entities, the entity value could be included:
          // #bot
          // Hello Me, my name is Bot!
          // ENTITIES NAME
          // ENTITY_VALUES Bot|Me
          // we could use the confidence of 'Bot' here, but we use the first one.
          const findAndDeleteFirstActualEntity = (name) => {
            for (let i = 0; i < actualEntitiesTemp.length; i++) {
              if (actualEntitiesTemp[i].name === name) {
                const found = actualEntitiesTemp[i]
                actualEntitiesTemp.splice(i, 1)
                return found
              }
            }
            return null
          }
          const expectedEntities = []
          for (let i = 0; i < expectedEntityNames.length; i++) {
            const name = expectedEntityNames[i]
            const entityKey = expectedEntityKeys[i]
            const lastValue = perExpectedEntity[name] || {
              name,
              count: 0,
              sum: 0,
              confidenceDifferenceSquaredSum: 0,
              avg: null,
              deviation: null
            }

            const actualEntity = findAndDeleteFirstActualEntity(name)
            if (actualEntity) {
              actualEntity.matches = true
            }
            const confidenceReal = actualEntity ? actualEntity.confidence : null
            const confidenceCorrected = _.isNil(confidenceReal) ? 0 : actualEntity.confidence
            expectedEntities.push({ name, confidenceReal, confidenceCorrected, matches: !!actualEntity, entityKey })

            // entity confidence will be null if chatbot does not returns condfidence, or there are no succesfuly resolved entities
            perExpectedEntity[name] = {
              name: lastValue.name,
              count: lastValue.count + 1,
              sum: _addSafe(lastValue.sum, confidenceCorrected),
              confidenceDifferenceSquaredSum: 0,
              avg: null,
              deviation: null
            }
          }

          // perActualEntity
          for (const actualEntity of actualEntities) {
            const lastValue = perActualEntity[actualEntity.name] || {
              name: actualEntity.name,
              count: 0,
              sum: 0,
              confidenceDifferenceSquaredSum: 0,
              avg: null,
              deviation: null
            }

            // entity confidence will be null if chatbot does not returns confidence, or there are no succesfuly resolved entities
            perActualEntity[actualEntity.name] = {
              name: lastValue.name,
              count: lastValue.count + 1,
              sum: _addSafe(lastValue.sum, actualEntity.confidence),
              confidenceDifferenceSquaredSum: 0,
              avg: null,
              deviation: null
            }
          }

          overallStat.sum = _addSafe(overallStat.sum, confidenceCorrected)
          overallStat.sumReal = _addSafe(overallStat.sum, confidenceReal)

          // perExpectedIntent
          if (expectedIntent) {
            const perExpectedIntentField = perExpectedIntent[expectedIntent] ? perExpectedIntent[expectedIntent] : {
              name: expectedIntent,
              count: 0,
              sum: 0,
              confidenceDifferenceSquaredSum: 0,
              avg: null,
              deviation: null
            }

            perExpectedIntent[expectedIntent] = {
              name: expectedIntent,
              count: perExpectedIntentField.count + 1,
              sum: _addSafe(perExpectedIntentField.sum, confidenceCorrected),
              confidenceDifferenceSquaredSum: 0,
              avg: null,
              deviation: null
            }
          }

          // perCorrectedIntent
          const perCorrectedIntentField = perCorrectedIntent[correctedIntent] ? perCorrectedIntent[correctedIntent] : {
            name: correctedIntent,
            count: 0,
            sum: 0,
            confidenceDifferenceSquaredSum: 0,
            avg: null,
            deviation: null
          }

          perCorrectedIntentField.incomprehension = perCorrectedIntentField.incomprehension || correctedIncomprehensionIntent

          perCorrectedIntent[correctedIntent] = {
            name: correctedIntent,
            incomprehension: perCorrectedIntentField.incomprehension,
            count: perCorrectedIntentField.count + 1,
            sum: _addSafe(perCorrectedIntentField.sum, confidenceCorrected),
            confidenceDifferenceSquaredSum: 0,
            avg: null,
            deviation: null
          }

          // perActualIntent
          const perActualIntentField = perActualIntent[actualIntent] ? perActualIntent[actualIntent] : {
            name: actualIntent,
            count: 0,
            sum: null,
            confidenceDifferenceSquaredSum: 0,
            avg: null,
            deviation: null
          }
          if (perActualIntentField.count && perActualIntentField.incomprehension !== actualIncomprehensionIntent) {
            const corrected = perActualIntentField.incomprehension || actualIncomprehensionIntent
            debug(`Ambigous incomprehension flag. Collected is ${perActualIntentField.incomprehension}, actual is ${actualIncomprehensionIntent}. Using ${corrected}`)
            perActualIntentField.incomprehension = corrected
          } else {
            perActualIntentField.incomprehension = actualIncomprehensionIntent
          }
          perActualIntent[actualIntent] = {
            name: actualIntent,
            incomprehension: perActualIntentField.incomprehension,
            count: perActualIntentField.count + 1,
            sum: _addSafe(perActualIntentField.sum, confidenceReal),
            confidenceDifferenceSquaredSum: 0,
            avg: null,
            deviation: null
          }

          utteranceListProcessor.process(actualIncomprehensionIntent, actualIncomprehensionIncompUtterance)
          // collecting and returning data
          const scriptId = entry.testSetScript ? (entry.testSetScript.id || entry.testSetScript.name)
            : (entry.testSetRepository ? entry.testSetRepository.id
              : (entry.testSetFolder ? entry.testSetFolder.id
                : (entry.testSetDownloadLink ? entry.testSetDownloadLink.id
                  : (entry.testSetExcel ? entry.testSetExcel.id : null))))
          const newEntry = {
            id: step.step + ' - ' + scriptId,
            utterance,
            utteranceKey: utterance + '_' + expectedIntent,
            scriptId,
            step: step.step,
            testcaseName: entry.testcaseName,
            testCaseResultId: entry.id,
            // because of a fatal error (because wrong structured convo for example) this entry cant be used in stat
            intent: {
              expected: expectedIntent,
              // just name
              actual: actualIntent,
              incomprehension: actualIncomprehensionIntent,
              incomprehensionIncompUtterance: actualIncomprehensionIncompUtterance,
              confidenceCorrected,
              confidenceReal,
              actuals: actualIntents,
              confidenceDifferenceExpected: null,
              confidenceDifferenceActual: null,
              matches: expectedIntent === actualIntent,
              confusionProbability: (actualIntents && actualIntents.length > 1 && _.isNumber(actualIntents[0].confidence) && _.isNumber(actualIntents[1].confidence)) ? (1 - actualIntents[0].confidence + actualIntents[1].confidence) : null
            },
            entity: {
              // [{ name, confidenceReal, confidenceCorrected, matches, entityKey }]
              expected: expectedEntities,
              expectedNames: expectedEntityNames,
              expectedValues: expectedEntityValues,
              // [{ .., name, confidence, matchechs, entityKey }]
              actual: actualEntities,
              // ... is in expected
              acceptMoreEntities
            }
          }
          perUtterance.push(newEntry)
          if (!newEntry.id) {
            debug(`cant find testset id for ${JSON.stringify(entry)}`)
          }
        }
      }
    }
    if (!processedStepFound) {
      if (!processableStepFound) {
        ignoredTestScripts.push({
          resultId: entry.id,
          testcaseName: entry.testcaseName,
          err: 'Has no step to process (no bot section)'
        })
      } else {
        ignoredTestScripts.push({
          resultId: entry.id,
          testcaseName: entry.testcaseName,
          err: 'Failed to process any step'
        })
      }
      overallStat.testScriptsIgnored++
    }
  }
}

class UtteranceListProcessor {
  constructor () {
    this.testCaseProcessable = null
    this.testCaseTestSetSciptUtterances = null
    this.utteranceListIdToStruct = {}
  }

  setTestCaseResult (testCaseResult) {
    this.testCaseProcessable = testCaseResult.testSet.expandUtterancesIncomprehension
    this.testCaseTestSetSciptUtterances = (testCaseResult.testSetScript && testCaseResult.testSetScript.scriptType === 'SCRIPTING_TYPE_UTTERANCES') ? testCaseResult.testSetScript : null
    this.testSet = testCaseResult.testSet
  }

  process (actualIncomprehensionIntent, actualIncomprehensionIncompUtterance) {
    if (!this.testCaseProcessable) {
      return
    }
    if (!this.testCaseTestSetSciptUtterances) {
      return
    }
    const script = this.testCaseTestSetSciptUtterances
    if (!this.utteranceListIdToStruct[script.id || script.name]) {
      this.utteranceListIdToStruct[script.id || script.name] = {
        testSetId: this.testSet.id,
        testSetName: this.testSet.name,
        scriptId: script.id,
        scriptName: script.name,
        correct: 0,
        incorrect: 0
      }
    }

    this.utteranceListIdToStruct[script.id || script.name][(actualIncomprehensionIntent || actualIncomprehensionIncompUtterance) ? 'incorrect' : 'correct']++
  }

  getTrainerUtteranceList () {
    return Object.values(this.utteranceListIdToStruct).map(e => {
      e.recall = (e.correct + e.incorrect) ? e.correct / (e.correct + e.incorrect) : null
      return e
    })
  }
}

module.exports.process = async ({ testCaseResults = [], connectorFeatures = {} }) => {
  const context = {
    utteranceListProcessor: new UtteranceListProcessor(),
    result: {
      perExpectedIntent: {},
      perExpectedEntity: {},
      perActualIntent: {},
      perActualEntity: {},
      perCorrectedIntent: {},
      perUtterance: [],
      ignoredTestScripts: [],
      ignoredTestScriptSteps: [],
      overallStat: {
        sum: 0,
        avg: null,
        sumReal: null,
        avgReal: null,
        // Where there is at least one step, which is ignored or accepted
        testScriptsProcessed: 0,
        testScriptsIgnored: 0,

        steps: 0,
        // bot sections with error or without intent and entity asserter
        stepsIgnored: 0,
        stepsWithActualIntent: 0,
        stepsWithActualIntentAndConfidence: 0,
        stepsWithIntent: 0,
        stepsWithEntity: 0,
        stepsWithEntityValue: 0,

        incomprehensionIntents: 0,
        matchedIntents: 0,
        missedIntents: 0,
        missedIntentsSum: null,
        missedIntentsAvg: null,

        justExpectedEntities: 0,
        // Entity is in response, but we dont expect it
        justActualEntitiesChecked: 0,
        // Entity is in response, but we have no expectations
        justActualEntitiesNotChecked: 0,
        expectedAndActualEntities: 0,

        confidenceDifferenceSquaredSumExpected: 0,
        confidenceDifferenceSquaredSumActual: 0,
        // this is a strange flag. We cant say for sure that the entity confidence is not supported, so this flag is null or true
        intentResolutionSupported: null,
        intentConfidenceSupported: null,
        intentListSupported: null,
        entityResolutionSupported: null,
        entityConfidenceSupported: null,
        expectedIntentsSupported: null,
        expectedEntitiesSupported: null,

        accuracy: null,
        precision: null,
        recall: null,
        F1: null
      }
    }
  }

  testCaseResults.forEach((entry, index) => _processTestCaseResult(entry, index, context))

  const setSupportFlag = (name, value) => {
    if (_.isNull(value) || context.result.overallStat[name] === value) {
      return
    }
    if (!_.isNull(context.result.overallStat[name])) {
      debug(`Overriding support flag "${name}" value "${context.result.overallStat[name]}" with "${value}"`)
    }
    context.result.overallStat[name] = value
  }
  setSupportFlag('intentConfidenceSupported', connectorFeatures.intentConfidenceScore)
  setSupportFlag('intentResolutionSupported', connectorFeatures.intentResolution)
  setSupportFlag('intentListSupported', connectorFeatures.alternateIntents)

  setSupportFlag('entityConfidenceSupported', connectorFeatures.entityConfidenceScore)
  setSupportFlag('entityResolutionSupported', connectorFeatures.entityResolution)

  _calculateAggregatedStats(context)
  _calculateDeviation(context)
  _calculateConfusionMatrix(context)
  _countUnexpectedIntents(context)
  _calculateConfidenceDistibution(context)
  _calculateUnexpectedIntentByAlternativeList(context)
  _calculateConfidenceThreshold(context)

  const result = Object.assign({}, context.result, {
    perExpectedIntent: Object.values(context.result.perExpectedIntent),
    perActualIntent: Object.values(context.result.perActualIntent),
    perCorrectedIntent: Object.values(context.result.perCorrectedIntent),
    perExpectedEntity: Object.values(context.result.perExpectedEntity),
    perActualEntity: Object.values(context.result.perActualEntity),
    utteranceList: context.utteranceListProcessor.getTrainerUtteranceList()
  })

  result.overallStat.actualIntentCount = result.perActualIntent.length
  result.overallStat.actualIntentCountNoIncomprehension = result.perActualIntent.filter(i => !i.incomprehension).length
  result.overallStat.actualEntityCount = result.perActualEntity.length
  delete result.overallStat.missedIntentsSum
  delete result.overallStat.sumReal

  return result
}
