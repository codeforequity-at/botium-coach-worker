const assert = require('chai').assert
const language = require('../src/language')

describe('language', function () {
  it('should recognize language', async () => {
    const lang1 = await language.guessLanguage('this is some english text')
    assert.equal(lang1, 'en')
    const lang2 = await language.guessLanguage('cest la vie, mon ami')
    assert.equal(lang2, 'fr')
    const lang3 = await language.guessLanguage('das ist ein deutscher text')
    assert.equal(lang3, 'de')
  })
})
