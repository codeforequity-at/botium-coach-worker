import pandas as pd

def flatten_intents_list(intents):
    relevant_data = {"utterance": list(), "intent": list()}
    for i in range(len(intents)):
        current_intent = intents[i]["name"]
        for j in range(len(intents[i]["examples"])):
            relevant_data["utterance"].append(intents[i]["examples"][j])
            relevant_data["intent"].append(current_intent)
    return pd.DataFrame(relevant_data)
