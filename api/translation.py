import json
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate(translationRequest):
    print(json.dumps(translationRequest, indent=2))
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # it does not looks thread safe!
    tokenizer.src_lang = translationRequest['language_from']
    encoded_ar = tokenizer(translationRequest['sentence'], return_tensors="pt")
    print(encoded_ar)
    forced_bos_token_id = None
    if translationRequest['language_to'] in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id[translationRequest['language_to']]
    else:
        # convert en_GB to en_XX
        language_to = translationRequest['language_to'][0:3] + "XX"
        if language_to in tokenizer.lang_code_to_id:
            forced_bos_token_id = tokenizer.lang_code_to_id[language_to]

    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=forced_bos_token_id
    )
    print(generated_tokens)
    return {"translation": tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]}

if __name__ == '__main__':
    print(translate({
            "sentence": 'To meet',
            "language_from": "en_GB",
            "language_to": "de_DE"
        }))