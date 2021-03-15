import json
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

prefix_with_xx = ['en', 'es', 'fr', 'ja', 'nl', 'pt', 'tl']

accepted_lang = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT',
                 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK',
                 'tr_TR', 'vi_VN', 'zh_CN', 'af_ZA', 'az_AZ', 'bn_IN', 'fa_IR', 'he_IL', 'hr_HR', 'id_ID', 'ka_GE',
                 'km_KH', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN', 'pl_PL', 'ps_AF', 'pt_XX', 'sv_SE', 'sw_KE', 'ta_IN',
                 'te_IN', 'th_TH', 'tl_XX', 'uk_UA', 'ur_PK', 'xh_ZA', 'gl_ES', 'sl_SI']

# convert en_GB to en_XX
def normalize_lang(lang):
    if lang[0:2] in prefix_with_xx :
        return lang[0:2] + "_XX"
    return lang

def translate(translationRequest):
    language_from = normalize_lang(translationRequest['language_from'])
    language_to = normalize_lang(translationRequest['language_to'])

    if language_from not in accepted_lang:
        return 'Not supported language_from ' + language_from, 400

    if language_to not in accepted_lang:
        return 'Not supported language_to ' + language_from, 400

    print(json.dumps(translationRequest, indent=2))
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # it does not looks thread safe!
    tokenizer.src_lang = language_from
    encoded_ar = tokenizer(translationRequest['sentence'], return_tensors="pt")

    forced_bos_token_id = tokenizer.lang_code_to_id[language_to]

    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=forced_bos_token_id
    )
    return {"translation": tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]}
