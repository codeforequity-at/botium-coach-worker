def create_pinecone_index(CreatePineconeIndexRequest):
    """
        Creates a Pinecone index to upload embeddings to
        
        inputs: name (string) - name specified to call index on pinecone
                environment (string) - pincone environment where index is stored

        output: result - Dict with 2 keys: 
                        status  - confirms if index was successfully created or not (True/False)
                        message - contains string stating if it was successfully created or failed with failure message
    """
    index = CreatePineconeIndexRequest['name']
    pine_api_key = os.environ.get('PINECONE_API')
    pine_env = CreatePineconeIndexRequest['environment']
    try :
        pinecone.init(api_key=pine_api_key, environment=pine_env)
        pinecone.create_index(index, dimension=1536, metric='cosine', pods=1, replicas=1)
        result = {'status': True,
                  'message': "Successfully created index."}
    except Exception as error:
    # handle the exception
        result = {'status': False,
                  'message': "Failed: an exception occurred: {0}".format(error)}
    return result


def upload_factcheck_documents(UploadFactcheckDocumentRequest):
    """
        Uploads embeddings to Pinecone index.

        inputs: index (string) - name of pinecone index to store embeddings
                environment (string) - pincone environment where index is stored
                fileptah (string) - filepath of where documents to be uploaded are stored

        output: content - Dict with 2 keys: 
                        status - confirms if index was successfully uploaded or not (True/False)
                        message - contains string stating if documents were sucessfully uploaded or failed with failure message.
    """
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')
    embedding_model = "text-embedding-ada-002"

    index = UploadFactcheckDocumentRequest['index']
    pine_env = UploadFactcheckDocumentRequest['environment']
    filepath= UploadFactcheckDocumentRequest['filepath']

    pineindex=pinecone_init(pine_api_key,pine_env,index)
    content=document_upsert_pinecone(openai, embedding_model, pineindex, index, filepath)
    return content

def factcheck(factcheckRequest):
    """
        Fact checks a statment given the ground truth docs stored on pinecone index

        inputs: index (string) - name of index where embeddings of ground truth docs are stored on pinecone
                environment (string) - pincone environmnet where index is stored
                statement (string) - the statement to be fact checked  

        output: Dictionary with 3 keys: 
                status - confirms if statement is factually correct or not True/False)
                reasoning - contains reasons why we beleive statement is true or false
                fixed_statement - edited version of statement based on reasons 
    """
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')

    index = factcheckRequest['index']
    pine_env = factcheckRequest['environment']
    statement = factcheckRequest['statement']

    pineindex=pinecone_init(pine_api_key,pine_env,index)
    editor_responses, agreement_gates, status =editor(openai, statement, pineindex, index)
   
    result = {  'status': status,
                'reasoning': agreement_gates,
                'fixed_statement': editor_responses}
    return result