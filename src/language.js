const { Language } = require('node-nlp')
const LangAll = require('@nlpjs/lang-all')

const _capitalize = (str) => str ? str.charAt(0).toUpperCase() + str.slice(1) : null

const guessLanguage = async (text) => {
  const language = new Language()
  const guess = language.guess(text)
  return guess[0].alpha2
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
  removeStopwords,
  tokenize
}
