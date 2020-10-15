const franc = require('franc')
const { Language } = require('node-nlp')
const LangAll = require('@nlpjs/lang-all')

const _capitalize = (str) => str ? str.charAt(0).toUpperCase() + str.slice(1) : null
const language = new Language()

const guessLanguage = async (text) => {
  const allGuesses = franc.all(text) || []
  for (const [guessAlpha3] of allGuesses) {
    const l = language.languagesAlpha3[guessAlpha3]
    if (l) {
      return l.alpha2
    }
  }
  throw new Error('Failed to identify language')
}

const guessLanguageForIntents = async (intents) => {
  const text = intents && intents.length > 0 ? intents.slice(0, 5).reduce((agg, intent) => [...agg, ...intent.utterances.slice(0, 5)], []).join(' ') : ''
  return guessLanguage(text)
}

const removeStopwords = (tokens, lang) => {
  const moduleName = `Stopwords${_capitalize(lang)}`
  if (LangAll[moduleName]) {
    const stopwords = new LangAll[moduleName]()
    return tokens.filter(t => !stopwords.isStopword(t))
  } else {
    return tokens
  }
}

const tokenize = (utterance, lang) => {
  const moduleName = `Tokenizer${_capitalize(lang)}`
  if (LangAll[moduleName]) {
    const tokenizer = new LangAll[moduleName]()
    return tokenizer.tokenize(utterance, true)
  } else {
    return utterance.split(' ')
  }
}

module.exports = {
  guessLanguage,
  guessLanguageForIntents,
  removeStopwords,
  tokenize
}
