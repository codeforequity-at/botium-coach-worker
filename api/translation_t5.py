import logging

# default translate pipeline uses t5, and support this languages:
accepted_language_from = ['en']
accepted_language_to = ['de', 'fr', 'ro']
from transformers import pipeline
from cachetools import cached, TTLCache

# ttl: ten minutes
@cached(cache=TTLCache(maxsize=10, ttl=600))
def get_pipeline(language_from, language_to):
    logging.debug(f'Initializing t5 model for "{language_from}" to "{language_to}"')
    return pipeline("translation_{language_from}_to_{language_to}".format(language_from = language_from, language_to = language_to))

def translate(translationRequest):
    language_from = translationRequest['language_from'][0:2]
    language_to = translationRequest['language_to'][0:2]
    if language_from not in accepted_language_from:
        return {'error': ['Not supported language_from ' + language_from, 400]}

    if language_to not in accepted_language_to:
        return {'error': ['Not supported language_to ' + language_to, 400]}

    return {"provider": "t5", "translation": get_pipeline(language_from, language_to)(translationRequest['sentence'])[0]['translation_text']}
