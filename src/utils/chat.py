from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.context_loader import *
from exception.operationhandler import system_logger, userops_logger, llmresponse_logger
import uuid
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.runnables import RunnablePassthrough
import torch

class Chat:
    def __init__(self,
                model:str,
                temperature:float,
                chunk_size:int = 3500,
                chunk_overlap:int = 0):
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = ChatGroq(api_key = groq_key,model = self.model, temperature= self.temperature)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_logger = system_logger
        self.userops_logger = userops_logger
        self.llm_logger = llmresponse_logger
        
    
    def create_contextual_chunks(self,file_path: str):
        
        self.system_logger.info(f"creating contextual chunks for {file_path}")
        try:
            loader = PyMuPDFLoader(file_path)
            doc_pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            doc_chunks = splitter.split_documents(doc_pages)
            original_doc = '\n'.join([doc.page_content for doc in doc_chunks])

            contextual_chunks = []
            for chunk in doc_chunks:
                chunk_content = chunk.page_content
                chunk_metadata = chunk.metadata
                chunk_metadata_upd = {
                    "id" : str(uuid.uuid4()),
                    "page": chunk_metadata["page"],
                    "source": chunk_metadata["source"],
                    "title": chunk_metadata["source"].split("/")[-1]
                }
                context = generate_chunk_context(original_doc, chunk_content)
                contextual_chunks.append(Document(page_content=context+'\n'+chunk_content,
                                                metadata = chunk_metadata_upd))
            self.system_logger.info(f"Created {len(contextual_chunks)} chunks")
            return contextual_chunks
        
        except Exception as e:
            self.system_logger.error(f"Error creating contextual chunks: {str(e)}")
            raise
    
    @property
    def get_embeddings(self):
        try:
            self.system_logger.info("initializing embedding model")
            model_name = "BAAI/bge-small-en-v1.5"
            model_kwargs = {'device': self.device}
            encode_kwargs = {'normalize_embeddings': True}
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            self.system_logger.info("Embedding model initialized successfully")
        except Exception as e:
            self.system_logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
        
        return self._embeddings
    
    def get_retriever(self, file_path: str, method: str = "hybrid") -> any:
        """Get document retriever with logging."""
        self.system_logger.info(f"Setting up {method} retriever")
        try:
            chunks = self.create_contextual_chunks(file_path)
            
            if method == "similarity":
                retriever = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.get_embeddings,
                    collection_name="similarity_collection",
                    collection_metadata={"hnsw:space": "cosine"}
                ).as_retriever(search_kwargs={"k": 5})
                
            elif method == "bm25":
                retriever = BM25Retriever.from_documents(documents=chunks, k=5)
                
            elif method == "hybrid":
                similarity_retriever = self.get_retriever(file_path, "similarity")
                bm25_retriever = self.get_retriever(file_path, "bm25")
                
                retriever = EnsembleRetriever(
                    retrievers=[similarity_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                
            elif method == "rerank":
                reranker = CrossEncoderReranker(
                    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"),
                    top_n=5
                )
                
                retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=self.get_retriever(file_path, "hybrid")
                )
            
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
            
            self.system_logger.info(f"{method} retriever setup complete")
            return retriever
            
        except Exception as e:
            self.system_logger.error(f"Error setting up {method} retriever: {str(e)}")
            raise
    

    def create_qa_chain(self, file_path: str, retrieval_method: str = "rerank"):
        """Create QA chain with logging."""
        self.system_logger.info("Creating QA chain")
        try:
            retriever = self.get_retriever(file_path, retrieval_method)
            
            prompt = """You are an expert in question-answering tasks.
                        Answer the following question using only the provided context.
                        If the answer cannot be found in the context, state that you don't know.
                        Provide detailed, well-formatted answers based on the context.

                        Question: {question}

                        Context: {context}

                        Answer:"""
            
            prompt_template = ChatPromptTemplate.from_template(prompt)
            
            qa_chain = (
                {
                    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | self.llm
            )
            
            self.system_logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            self.system_logger.error(f"Error creating QA chain: {str(e)}")
            raise

    def generate_response(self, qa_chain, question: str):
        """Process a query with logging."""
        self.userops_logger.info(f"Processing query: {question}")
        try:
            response = qa_chain.invoke(question)
            self.llm_logger.info(f"Question: {question}\nResponse: {response}")
            return response
        except Exception as e:
            self.system_logger.error(f"Error processing query: {str(e)}")
            raise

   