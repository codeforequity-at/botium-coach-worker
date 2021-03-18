MODEL_MBART_LARGE_50 = 'facebook/mbart-large-50-many-to-many-mmt'
MODEL_T5 = 't5'
from . import translation_mbart_large_50
from . import translation_t5


def translate(translationRequest):
    guess_model = ('model' not in translationRequest) or (translationRequest['model'] is None)

    if guess_model or translationRequest['model'] == MODEL_T5:
        result = translation_t5.translate(translationRequest)
        if 'error' in result:
            if 'model' in translationRequest and translationRequest['model'] == MODEL_T5:
                return result['error']
        else:
            return result

    if guess_model or translationRequest['model'] == MODEL_MBART_LARGE_50:
        result = translation_mbart_large_50.translate(translationRequest)
        if 'error' in result:
            if 'model' in translationRequest and translationRequest['model'] == MODEL_MBART_LARGE_50:
                return result['error']
        else:
            return result

    if guess_model:
        return 'No module found supporting language_from "{language_from}", language_to "{language_to}"!'.format(
            language_from=translationRequest['language_from'], language_to=translationRequest['language_to']), 400

    return 'Module "{model}" not found!'.format(model = translationRequest['model']), 400
