from ast import Return
import json
import math
import numpy as np
import openai
import os
import pandas as pd
import pickle
import pinecone
import PyPDF2
import re
import requests
import sys
import time
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import torch

def create_query(openai,response_llm):
    """ Create query/facts which are required to be verified

        Input:  openai (class) - OpenAI API client 
                response_llm (string) - statement to be fact checked

        Output: questions (List) - list of questions to gather evidence to fact check statement
    """
    response_llm='Statement:= '+ response_llm
    print(response_llm)
    # Creating queries from the statement 
    print('-------------------------------------------------------------------------------------------------------')
    response=openai.ChatCompletion.create(
      model="gpt-3.5-turbo",messages=[
            {"role": "system", "content": "You are a helpful assistant with the ability to verify the facts in a given statement. Your task is to read the provided statement and break it down into individual facts, sentences, or contexts that require verification. Each aspect of the statement should be treated with a level of skepticism, assuming that there might be some factual errors. Your role is to generate queries to validate each fact, seeking clarification to ensure accurate and consistent information. Please assist in fact-checking by asking questions to verify the details presented in the statement."},
            {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd"},
            {"role": "assistant", "content": "Query generation \n Verify:= 1.Who sings the song Time of My Life? \n Verify:= 2.Is the song writer American?\n Verify:= 3.Which year the song was sung?\n Verify:= 4.Which film is the song Time of My Life from? \n Verify:= 5.Who produced the song Time of My Life?"},
            {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
            {"role": "assistant", "content": "Query generation \n Verify:= 1.Does your nose switch between nostrils? \n Verify:= 2.How often does your nostrils switch? \n Verify:= 3.Why does your nostril switch? \n Verify:= 4.What is nasal cycle?"},
          {"role": "user", "content":response_llm }
        ]
        )
    
    api_response=response['choices'][0]['message']['content']
    questions = []
    search_string='Verify'
    for question in api_response.split("\n"):
            # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)
        
    return questions
# gets context passages from the pinecone index
def get_context(question, index, indexname, top_k):
    """ Generate embeddings for the question

        Input:  question (string) - question used to gather evidence on statement from documents stored on pinecone
                index (class) - Pinecone API client
                indexname (string) - name of index used to store embeddings in pinecone
                top_k (int) - sets The number of results to return for each query

        Output: context (dict) - returns most relevant contect based on questions asked and retrieval score from pineocne
    """
    result = openai.Embedding.create(model="text-embedding-ada-002",input=question)
    embedding=result["data"][0]["embedding"]
    # search pinecone index for context passage with the answer
    context = index.query(namespace=indexname, vector = embedding, top_k=top_k, include_metadata=True)
    return context


# For each question retrieve the most relvant part of document 
def retrieval_passage(openai, response_llm, pineindex, indexname) :
    """ Generate embeddings for the question

        Input:  openai (class) - OpenAI API client 
                response_llms (string) - statement to be fact checked
                pineindex (class) - Pinecone API client
                indexname (string) - name of index specified by user to be created


        Output: used_evidences (list) - list of evidence to support if statement is true or false
    """
    questions=create_query(openai,response_llm)
    used_evidences=[]
    if len(questions)==0:
        return used_evidences
    
    print('Queries Created from the statement',questions)
    print('-------------------------------------------------------------------------------------------------------')
    query_search = []
    for query in questions:
        print('Retrieving relevant passage for query:', query )
        retrieved_passages = []
        #gets context passages from the pinecone index
        context = get_context(query, pineindex, indexname, top_k=1)

        for passage in context["matches"]:
               retrieved_passages.append(
                   {
                       "text": passage['metadata']['context'],
                       "query": query,
                       "retrieval_score": passage['score']
                   }
               )

        print(retrieved_passages) 
        ## figure conflicting articles and stop 
        if retrieved_passages:
            # Sort all retrieved passages by the retrieval score.
            retrieved_passages = sorted(
                retrieved_passages, key=lambda d: d["retrieval_score"], reverse=True
            )

            # Normalize the retreival scores into probabilities
            scores = [r["retrieval_score"] for r in retrieved_passages]
            probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
            for prob, passage in zip(probs, retrieved_passages):
                passage["score"] = prob
        query_search.append(retrieved_passages)
        
    used_evidences=[e for cur_evids in query_search for e in cur_evids[:1]]
    return used_evidences

def agreement_gate(openai,response_llm,pineindex, indexname):
    """ Generate embeddings for the question

        Input:  openai (class) - OpenAI API client 
                response_llms (string) - statement to be fact checked
                pineindex (class) - Pinecone API client
                indexname (string) - name of index specified by user to be created

        Output: agreement_gates (list) - contains reasoning, decision and is_open flag of query based on statemment  
                used_evidences (list) - list of evidence to support if statment is true or false
                relevance (int) - shows relevance of passage 
    """

    #Calling retrieval stage before agreement stage
    used_evidences=retrieval_passage(openai,response_llm, pineindex, indexname)
    agreement_responses=[]
    agreement_gates=[]
    
    # Checking relevant articles are present or not in the dataset provided
    relevance=0
    
    print('\n')
    print('-------------------------------------------------------------------------------------------------------')
    print('Evidences gathered for each query we are fact checking ')
    print('*************************************************')
    for i in used_evidences:
        print(i)
        print('*************************************************')
    print('-------------------------------------------------------------------------------------------------------')
    print('\n')
    # No evidence then return empty
    if len(used_evidences)==0:
        return agreement_gates,used_evidences,relevance
    
    for i in range(len(used_evidences)):
        if used_evidences[i]['retrieval_score']<0:
            relevance+=1
            
    if relevance >0:
        return agreement_gates,used_evidences,relevance
    
    for i in range(len(used_evidences)):
        user_llm=  "Statement:= " + response_llm + " \n Query:= " + used_evidences[i]['query'] + " \n Article:= " + used_evidences[i]['text']
        response=openai.ChatCompletion.create(
                  model="gpt-4",
                  messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in performing fact-checking between a given statement and an accompanying document based on the queries provided. Your goal is to ensure consistent and accurate results throughout the fact-checking process. For each query, you will compare both the statement and the document to determine if they agree or disagree on the specific facts presented. Any even slight agreement or disagreement between the two will be concluded as disagree. You will thoroughly provide reasoning for each conclusion reached and in therefore explicilty tell if you agree or disagree. If there are any discrepancies or inconsistencies between the statement and the article you will explicitly state disagree for clarity."  },
                {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. \n Query:= Who was the producer of (I’ve Had) The Time of My Life? \n Article:= On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University."},
                {"role": "assistant", "content": "Reasoning:= The article said that a demo was produced by Michael Lloyd and you said Time of My Life was produced by Michael Lloyd. \n Therefore:= This agrees with statement claims."},
                {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle. \n Query:= How often do your nostrils switch? \n Article:= Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One."},
                {"role": "assistant", "content": "Reasoning:= The article said the nose’s switching time is about every 2 hours, and you said the nose's switching time is about every 45 minutes. \n Therefore:= This disagrees with statement claims."},
                {"role": "user", "content":user_llm }])
        agreement_responses.append(response)
        
    
    for i in range(len(agreement_responses)):
        api_response=agreement_responses[i]['choices'][0]['message']['content'].strip().split("\n")
        if len(api_response)<2:
            reason = "Failed to parse."
            decision = None
            is_open = False
        else:
            reason = api_response[0]
            decision = api_response[1].split("Therefore:")[-1].strip()
            is_open = "disagrees" in api_response[1]
            gate = {"is_open": is_open, "reason": reason, "decision": decision}
            agreement_gates.append(gate)
    return agreement_gates,used_evidences,relevance


def editor(openai,response_llm, pineindex, indexname):
    """
    Create the Pinecone index

    Inputs: openai (class) - OpenAI API client 
            response_llms (string) - statement to be fact checked
            pineindex (string) - region where index is to be stored
            indexname (string) - name of index specified by user to be created

    Output: edited_response (string) - statement if edited ot not
            agreeemnet_gate (dict) - contains reason, decisions and gate value 
            status (boolean) - returns True if statement has passed fact check  
    """
    agreement_gates,used_evidences,relevance=agreement_gate(openai,response_llm, pineindex, indexname)
    edit_count=0
    edited_responses=[]
    
    if len(agreement_gates)==0 and len(used_evidences)==0 and relevance==0:
        print('Not enough data in the statement for performing fact checking')
        return edited_responses, agreement_gates, False
    
    if relevance == len(used_evidences):
        print('There is no document which is relevant to any/some of the facts present in statement')
        return edited_responses, agreement_gates, False
    
    
    print('-------------------------------------------------------------------------------------------------------')
    print('Agreement gate for each query if the statement agrees or not')
    print('*************************************************')
    for i in agreement_gates:
        print(i)
        print('*************************************************')
    print('-------------------------------------------------------------------------------------------------------')
    print('\n')
    for i in range(len(agreement_gates)):
        if agreement_gates[i]['is_open']:
            user_llm=  "Statement:= " + response_llm + " \n Query:= " + used_evidences[i]['query'] + " \n Article:= " + used_evidences[i]['text'] + agreement_gates[i]['reason']
            response=openai.ChatCompletion.create(
                          model="gpt-4",
                          messages=[
                        #{"role": "system", "content": "You are a helpful assistant.Who fixes the statement using the reasoning provided as there is a disagreement between article and statement on the query."  },
                        {"role": "system", "content": "You are a helpful assistant specializing in fixing statements based on the provided reasoning when there is a disagreement between the statement and the provided documentation on the given query. Your objective is to ensure consistent and accurate results by modifying the facts in the statement using information from the accompanying documentation, guided by the understanding provided in the reasoning."  },#Just modify the facts don't try to add any new information, try to keep as close as possible to the original statement
                        {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. \n Query:= Who was the producer of (I’ve Had) The Time of My Life? \n Article:= On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University. \n Reasoning:= Time of My Life producer name in your statement is wrong."},
                        {"role": "assistant", "content": "Fixed Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd."},
                        {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle. \n Query:= How often do your nostrils switch? \n Article:= Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One. \n Reasoning:= This suggests 45 minutes switch time in your statement is wrong."},
                        {"role": "assistant", "content": "Fixed Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
                        {"role": "user", "content":user_llm }])
            edit_count +=1
            edited_responses.append(response['choices'][0]['message']['content'])
            response_llm=response['choices'][0]['message']['content']
        
    if edit_count==0:
        print('Nothing to edit as the statement seems to be factually correct')
        edited_responses = "Sucessfully fact checked and statement appears to be correct"
        status = True
    else:
        print('Edited Statements Based of the disagreement of facts in documentation found and statement made')
        print('\n')
        print('*************************************************')
        for i in edited_responses:
            print(i)
            print('*************************************************')
        status = False

    return edited_responses, agreement_gates, status

# Function to initialize pinecone index
def pinecone_init(api_key,environment,index_name):
    """
    Create the Pinecone index

    Inputs: api_key (string) - Pinecone API key   
            environment (string) - region where index is to be stored
            index_name (string) - name of index specified by user to be created

    Output: document similarities - Dict of most relevant passages from documents 
    """
    try :
        pinecone.init(api_key=api_key, environment=environment )
        index = pinecone.Index(index_name)
        return index
    except Exception as error:
    # handle the exception
        print("An exception occurred:", error)

def document_preprocessing(text):
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
        print("Failed to preprocess documents: {0}".format(error))
        return False

def upsert_document(openai, split_content, filename, page_num, embedding_model, pineindex, indexname):
    """
       Inserts managable content into pinecone index

       Inputs:  openai (class) - OpenAI API client 
                split_content (lsit) - List of content split into small chunk sizes
                filename (string) - name fo file that is being processed
                page_num (int) - page number of documnet being processed
                embedding_model (string) - name of model used to create embeddings
                indexname (string) - Name of Pinecone index to store embeddings from documents

        output: None
    """
    para=0
    try:
        # Append the split content to the list
        for content in split_content:
            para +=1
            iid= filename[:-4]+'_' +str(page_num)+ '_'+str(para)
            result = openai.Embedding.create(model=embedding_model,input=content)
            embedding=result["data"][0]["embedding"]
            vector = [{'id': iid,
                    'values':embedding,
                    'metadata':{"filename": filename, "word_count": len(content.split()), 'context': content}
                    }]
            pineindex.upsert(vectors=vector, namespace=indexname ) 
            print('Uploaded content to Pinecone index. {0}'.format(vector))
    except Exception as error:
        raise Exception("Failed to upload content: {0}".format(error))

def document_upsert_pinecone(openai, embedding_model, pineindex, indexname, filepath):
    """
        Reads the documents and breaks into contents of managable size(300 words) and inserts into pinecone index

        Inputs: openai (class) - OpenAI API client
                embedding_model (string) -
                pineindex (class) - Pinecone API client
                indexname (string) - name of Pinecone index to store embeddings
                filepath (string) - path to where documents are located on EFS

        output: Dictionary with 2 keys: 
                status - confirms if documents were successfully uploaded to pinecone ot not (True/False)
                message - contains string stating if it was successfully uploaded or failed with failure message.
    """
    try:
        print('Processing documents....')
        for filename in os.listdir(filepath):
            if filename.endswith(".pdf"):
                # Open PDF file
                pdf_file = open(os.path.join(filepath, filename), "rb")
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Loop through pages and split into documents of 300 tokens
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    content = document_preprocessing(page_text)
                    upsert_document(openai, content, filename, page_num, embedding_model, pineindex, indexname)
                    
            elif filename.endswith(".txt"):    
                # Read text file
                with open(os.path.join(filepath, filename), "r") as f:
                    text = f.read()
                page_num = 0
                content = document_preprocessing(text)
                upsert_document(openai, content, filename, page_num, embedding_model, pineindex, indexname)

            result = {  'status': True,
                        'message': "Successfully processed and uploaded all documents"}
    except Exception as error:
        result = {  'status': False,
                    'message': "Failed: an exception occurred: {0}".format(error)}
    return result
