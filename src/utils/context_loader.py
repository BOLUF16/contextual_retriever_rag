from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
from config.appconfig import groq_key,hf_key

llm = ChatGroq(model="qwen-2.5-32b", temperature=0, api_key=groq_key)

def generate_chunk_context(document,chunk):
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
    
    prompt_template = ChatPromptTemplate.from_template(chunk_process_prompt)
    chunk_chain = (prompt_template | llm | StrOutputParser())
    context = chunk_chain.invoke({"document": document, "chunk": chunk})
    
    return context
