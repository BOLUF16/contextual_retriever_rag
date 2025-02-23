from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
from config.appconfig import groq_key,hf_key

llm = ChatGroq(model="qwen-2.5-32b", temperature=0, api_key=groq_key)

def generate_chunk_context(document, chunk):
    """
    Generates a concise context for a specific chunk of text within a larger document.

    This function uses a language model to analyze the document and the chunk, and then
    provides a brief context to situate the chunk within the overall document. The context
    is designed to improve search retrieval and understanding of the chunk.

    Args:
        document (str): The entire document as a string.
        chunk (str): The specific chunk of text for which context is needed.

    Returns:
        str: A concise context (3-4 sentences) for the chunk, situating it within the document.

    Example:
        >>> document = "This is a long document about AI and machine learning..."
        >>> chunk = "Machine learning algorithms are used for predictive modeling."
        >>> generate_chunk_context(document, chunk)
        "Focuses on the application of machine learning algorithms in predictive modeling."
    """
    # Define the prompt template for generating chunk context
    chunk_process_prompt = """You are an AI assistant specializing in document 
                              analysis. Your task is to provide brief, 
                              relevant context for a chunk of text based on the 
                              following document.

                              Here is the document:
                              <document>
                              {document}
                              </document>
                            
                              Here is the chunk we want to situate within the whole 
                              document:
                              <chunk>
                              {chunk}
                              </chunk>
                            
                              Provide a concise context (3-4 sentences max) for this 
                              chunk, considering the following guidelines:

                              - Give a short succinct context to situate this chunk 
                                within the overall document for the purposes of  
                                improving search retrieval of the chunk.
                              - Answer only with the succinct context and nothing 
                                else.
                              - Context should be mentioned like 'Focuses on ....'
                                do not mention 'this chunk or section focuses on...'
                              
                              Context:
                           """
    
    # Create a prompt template from the defined prompt
    prompt_template = ChatPromptTemplate.from_template(chunk_process_prompt)
    
    # Define the chain for processing the prompt and generating the context
    chunk_chain = (prompt_template | llm | StrOutputParser())
    
    # Invoke the chain with the document and chunk as inputs
    context = chunk_chain.invoke({"document": document, "chunk": chunk})
    
    # Return the generated context
    return context
