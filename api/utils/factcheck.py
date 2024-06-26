import openai
from pinecone import Pinecone
import re
import torch
from .retry import openai_retry_with_exponential_backoff

@openai_retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@openai_retry_with_exponential_backoff
def embeddings_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)

def create_query(openai, response_llm):
    """ Create query/facts which are required to be verified

        Input:  openai (class) - OpenAI API client 
                response_llm (string) - statement to be fact checked

        Output: questions (List) - list of questions to gather evidence to fact check statement
    """
    response_llm = 'Statement:= ' + response_llm
    response = completions_with_backoff(
        model="gpt-4", messages=[
            {"role": "system", "content": "You are a helpful assistant with the ability to verify the facts in a given statement. Your task is to read the provided statement and break it down into individual facts, sentences, or contexts that require verification. Each aspect of the statement should be treated with a level of skepticism, assuming that there might be some factual errors. Your role is to generate queries to validate each fact, seeking clarification to ensure accurate and consistent information. Please assist in fact-checking by asking questions to verify the details presented in the statement."},
            {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd"},
            {"role": "assistant", "content": "Query generation \n Verify:= 1.Who sings the song Time of My Life? \n Verify:= 2.Is the song writer American?\n Verify:= 3.Which year the song was sung?\n Verify:= 4.Which film is the song Time of My Life from? \n Verify:= 5.Who produced the song Time of My Life?"},
            {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
            {"role": "assistant", "content": "Query generation \n Verify:= 1.Does your nose switch between nostrils? \n Verify:= 2.How often does your nostrils switch? \n Verify:= 3.Why does your nostril switch? \n Verify:= 4.What is nasal cycle?"},
            {"role": "user", "content": response_llm}
        ]
    )

    api_response = response['choices'][0]['message']['content']
    questions = []
    search_string = 'Verify:='
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string, 1)[-1].strip()
        question = question.split('.', 1)[-1].strip()
        if question is not None and len(question) > 0:
            questions.append(question)
    return questions


def create_sample_questions(openai, text):
    text = 'Statement:= ' + text
    response = completions_with_backoff(
        model="gpt-4", messages=[
            {"role": "system", "content": "You are a helpful assistant with the ability to verify the facts in a given statement. Your task is to read the provided statement and break it down into individual facts, sentences, or contexts that require verification. Each aspect of the statement should be treated with a level of skepticism, assuming that there might be some factual errors. Your role is to generate queries to validate each fact, seeking clarification to ensure accurate and consistent information. Please assist in fact-checking by asking questions to verify the details presented in the statement."},
            {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd"},
            {"role": "assistant", "content": "Query generation \n Question:= Who sings the song Time of My Life? \n Question:= Is the song writer American?\n Question:= Which year the song was sung?\n Question:= Which film is the song Time of My Life from? \n Question:= Who produced the song Time of My Life?"},
            {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
            {"role": "assistant", "content": "Query generation \n Question:= Does your nose switch between nostrils? \n Question:= How often does your nostrils switch? \n Question:= Why does your nostril switch? \n Question:= What is nasal cycle?"},
            {"role": "user", "content": text}
        ]
    )

    api_response = response['choices'][0]['message']['content']
    questions = []
    search_string = 'Question:='
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string, 1)[-1].strip()
        questions.append(question)

    return questions

# gets context passages from the pinecone index


def get_context(question, index, namespace, top_k):
    """ Generate embeddings for the question

        Input:  question (string) - question used to gather evidence on statement from documents stored on pinecone
                index (class) - Pinecone API client
                namespace (string) - name of index used to store embeddings in pinecone
                top_k (int) - sets The number of results to return for each query

        Output: context (dict) - returns most relevant contect based on questions asked and retrieval score from pineocne
    """
    result = embeddings_with_backoff(model="text-embedding-ada-002", input=question)
    embedding = result["data"][0]["embedding"]
    # search pinecone index for context passage with the answer
    context = index.query(namespace=namespace, vector=embedding,
                          top_k=top_k, include_metadata=True)
    return context


# For each question retrieve the most relvant part of document
def retrieval_passage(logger, openai, response_llm, pineindex, namespace):
    """ Generate embeddings for the question

        Input:  openai (class) - OpenAI API client 
                response_llms (string) - statement to be fact checked
                pineindex (class) - Pinecone API client
                namespace (string) - name of index specified by user to be created


        Output: used_evidences (list) - list of evidence to support if statement is true or false
    """
    questions = create_query(openai, response_llm)
    used_evidences = []
    if len(questions) == 0:
        return used_evidences

    logger.info(f'Queries Created from the statement: {",".join(questions)}')
    query_search = []
    for query in questions:
        logger.info(f'Retrieving relevant passage for query: {query}')
        retrieved_passages = []
        # gets context passages from the pinecone index
        context = get_context(query, pineindex, namespace, top_k=1)

        for passage in context["matches"]:
            retrieved_passages.append(
                {
                    "text": passage['metadata']['context'],
                    "query": query,
                    "retrieval_score": passage['score']
                }
            )

        logger.info(retrieved_passages)
        # figure conflicting articles and stop
        if retrieved_passages:
            # Sort all retrieved passages by the retrieval score.
            retrieved_passages = sorted(retrieved_passages, key=lambda d: d["retrieval_score"], reverse=True)

            # Normalize the retreival scores into probabilities
            scores = [r["retrieval_score"] for r in retrieved_passages]
            probs = torch.nn.functional.softmax(
                torch.Tensor(scores), dim=-1).tolist()
            for prob, passage in zip(probs, retrieved_passages):
                passage["score"] = prob
        query_search.append(retrieved_passages)

    used_evidences = [e for cur_evids in query_search for e in cur_evids[:1]]
    return used_evidences


def agreement_gate(logger, openai, response_llm, pineindex, namespace):
    """ Generate embeddings for the question

        Input:  openai (class) - OpenAI API client 
                response_llms (string) - statement to be fact checked
                pineindex (class) - Pinecone API client
                namespace (string) - name of index specified by user to be created

        Output: agreement_gates (list) - contains reasoning, decision and is_open flag of query based on statemment  
                used_evidences (list) - list of evidence to support if statment is true or false
                relevance (int) - shows relevance of passage 
    """

    # Calling retrieval stage before agreement stage
    used_evidences = retrieval_passage(logger,
        openai, response_llm, pineindex, namespace)
    agreement_responses = []
    agreement_gates = []

    # Checking relevant articles are present or not in the dataset provided
    relevance = 0

    logger.info('Evidences gathered for each query we are fact checking ')
    for i in used_evidences:
        logger.info(i)
    logger.info('*************************************************')

    # No evidence then return empty
    if len(used_evidences) == 0:
        return agreement_gates, used_evidences, relevance

    for i in range(len(used_evidences)):
        if used_evidences[i]['retrieval_score'] < 0:
            relevance += 1

    if relevance > 0:
        return agreement_gates, used_evidences, relevance

    for i in range(len(used_evidences)):
        user_llm = "Statement:= " + response_llm + " \n Query:= " + \
            used_evidences[i]['query'] + \
            " \n Article:= " + used_evidences[i]['text']
        response = completions_with_backoff(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in fact-checking, ensuring consistency and accuracy. Your task is to compare statements with accompanying documents, determining agreement or disagreement on specific facts. Even slight discrepancies will result in disagreement. Provide clear reasoning for each conclusion, explicitly stating agreement or disagreement.Do not rely on prior knowledge other than what is in the document."  },
                {"role": "user",
                 "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. \n Query:= Who was the producer of (I’ve Had) The Time of My Life? \n Article:= On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University."},
                {"role": "assistant", "content": "Reasoning:= The article said that a demo was produced by Michael Lloyd and you said Time of My Life was produced by Michael Lloyd. \n Therefore:= This agrees with statement claims."},
                {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle. \n Query:= How often do your nostrils switch? \n Article:= Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One."},
                {"role": "assistant", "content": "Reasoning:= The article said the nose’s switching time is about every 2 hours, and you said the nose's switching time is about every 45 minutes. \n Therefore:= This disagrees with statement claims."},
                {"role": "user", "content": user_llm}])
        agreement_responses.append(response)

    for i in range(len(agreement_responses)):
        api_response = agreement_responses[i]['choices'][0]['message']['content'].strip().split("\n")
        if len(api_response) < 2:
            reason = "Failed to parse."
            decision = None
            is_open = False
        else:
            reason = api_response[0].split("Reasoning:=", 1)[-1].strip()
            decision = api_response[1].split("Therefore:=", 1)[-1].strip()
            is_open = "disagrees" in api_response[1]
            gate = {"is_open": is_open, "reason": reason, "decision": decision}
            agreement_gates.append(gate)
    return agreement_gates, used_evidences, relevance


def editor(logger, openai, response_llm, pineindex, namespace):
    """
    Create the Pinecone index

    Inputs: openai (class) - OpenAI API client 
            response_llms (string) - statement to be fact checked
            pineindex (string) - region where index is to be stored
            namespace (string) - name of index specified by user to be created

    Output: edited_response (string) - statement if edited ot not
            agreeemnet_gate (dict) - contains reason, decisions and gate value 
            status (boolean) - returns True if statement has passed fact check  
    """
    agreement_gates, used_evidences, relevance = agreement_gate(logger,
        openai, response_llm, pineindex, namespace)
    edit_count = 0
    edited_responses = []

    if len(agreement_gates) == 0 and len(used_evidences) == 0 and relevance == 0:
        logger.info('Not enough data in the statement for performing fact checking')
        return edited_responses, agreement_gates, False

    if relevance == len(used_evidences):
        logger.info('There is no document which is relevant to any/some of the facts present in statement')
        return edited_responses, agreement_gates, False

    logger.info('Agreement gate for each query if the statement agrees or not')
    for i in agreement_gates:
        logger.info(i)
    logger.info('*************************************************')
    for i in range(len(agreement_gates)):
        if agreement_gates[i]['is_open']:
            user_llm = "Statement:= " + response_llm + " \n Query:= " + \
                used_evidences[i]['query'] + " \n Article:= " + \
                used_evidences[i]['text'] + agreement_gates[i]['reason']
            response = completions_with_backoff(
                model="gpt-4",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant.Who fixes the statement using the reasoning provided as there is a disagreement between article and statement on the query."  },
                    # Just modify the facts don't try to add any new information, try to keep as close as possible to the original statement
                    {"role": "system", "content": "You are a helpful assistant specializing in fixing statements based on the provided reasoning when there is a disagreement between the statement and the provided documentation on the given query. Your objective is to ensure consistent and accurate results by modifying the facts in the statement using information from the accompanying documentation, guided by the understanding provided in the reasoning."},
                    {"role": "user",
                        "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. \n Query:= Who was the producer of (I’ve Had) The Time of My Life? \n Article:= On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University. \n Reasoning:= Time of My Life producer name in your statement is wrong."},
                    {"role": "assistant", "content": "Fixed Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd."},
                    {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle. \n Query:= How often do your nostrils switch? \n Article:= Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One. \n Reasoning:= This suggests 45 minutes switch time in your statement is wrong."},
                    {"role": "assistant", "content": "Fixed Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
                    {"role": "user", "content": user_llm}])
            edit_count += 1
            api_response = response['choices'][0]['message']['content']
            edited_responses.append(api_response.split("Fixed Statement:=", 1)[-1].strip())
            response_llm = api_response

    if edit_count == 0:
        logger.info('Nothing to edit as the statement seems to be factually correct')
        edited_responses = "Successfully fact checked and statement appears to be correct"
        status = True
    else:
        logger.info('Edited Statements Based of the disagreement of facts in documentation found and statement made')
        for i in edited_responses:
            logger.info(i)

        logger.info('*************************************************')
        status = False

    return edited_responses, agreement_gates, status

# Function to initialize pinecone index


def pinecone_init(logger, api_key, environment, index_name):
    """
    Create the Pinecone index

    Inputs: api_key (string) - Pinecone API key   
            environment (string) - region where index is to be stored
            index_name (string) - name of index specified by user to be created

    Output: document similarities - Dict of most relevant passages from documents 
    """
    pc = Pinecone(api_key=api_key, environment=environment)
    index = pc.Index(index_name)        
    return index

def document_preprocessing(logger, text):
    """
       Breaks up texts into contents of managable size (300 words) to uplaod to Pinecone

       Inputs:  text (string) - textual data from document to be preprocessed
       Outputs: split_content (list) - content from texts passed as input split into manageable chunks sizes  
    """
    split_content = []
    try:
        current_content = ""
        current_word_count = 0
        for sentence in re.split("(?<=[.!?]) +", text):
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count > 300:
                if current_content != "":
                    split_content.append(current_content.strip())
                current_content = sentence
                current_word_count = sentence_word_count
            else:
                current_content += " " + sentence
                current_word_count += sentence_word_count

        if current_content != "":
            split_content.append(current_content.strip())
        return split_content
    except Exception as error:
        logger.info("Failed to preprocess documents: {0}".format(error))
        return False


def upsert_document(logger, openai, split_content, filename, page_num, embedding_model, pineindex, namespace):
    """
       Inserts managable content into pinecone index

       Inputs:  openai (class) - OpenAI API client 
                split_content (lsit) - List of content split into small chunk sizes
                filename (string) - name fo file that is being processed
                page_num (int) - page number of documnet being processed
                embedding_model (string) - name of model used to create embeddings
                namespace (string) - Name of Pinecone namespace to store embeddings from documents

        output: None
    """
    para = 0
    try:
        # Append the split content to the list
        for content in split_content:
            para += 1
            iid = filename[:-4] + '_' + str(page_num) + '_' + str(para)
            result = embeddings_with_backoff(model=embedding_model, input=content)
            embedding = result["data"][0]["embedding"]
            vector = [{'id': iid,
                       'values': embedding,
                       'metadata': {"filename": filename, "word_count": len(content.split()), 'context': content}
                       }]
            pineindex.upsert(vectors=vector, namespace=namespace)
            logger.info('Uploaded content to Pinecone index. {0} {1}'.format(iid, vector[0]["metadata"]))
    except Exception as error:
        raise Exception("Failed to upload content: {0}".format(error))


def document_upsert_pinecone(logger, openai, embedding_model, pineindex, namespace, filename, text):
    """
        Reads the documents and breaks into contents of managable size(300 words) and inserts into pinecone index

        Inputs: openai (class) - OpenAI API client
                embedding_model (string) -
                pineindex (class) - Pinecone API client
                namespace (string) - name of Pinecone index to store embeddings
                filename (string) - filename
                text (string) - text content

        output: Dictionary with 2 keys: 
                status - confirms if documents were successfully uploaded to pinecone ot not (True/False)
                message - contains string stating if it was successfully uploaded or failed with failure message.
    """
    try:
        logger.info('Processing documents....')
        content = document_preprocessing(logger, text)
        upsert_document(logger, openai, content, filename, 0,
                        embedding_model, pineindex, namespace)

        result = {'status': True,
                  'message': "Successfully processed and uploaded all documents"}
    except Exception as error:
        result = {'status': False,
                  'message': "Failed: an exception occurred: {0}".format(error)}
    return result
