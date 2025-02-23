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
                model: str,
                temperature: float,
                chunk_size: int = 3500,
                chunk_overlap: int = 0):
        """
        Initializes the Chat class with the specified model, temperature, and text chunking parameters.

        Args:
            model (str): The name of the language model to use.
            temperature (float): Controls the randomness of the model's output. Higher values mean more randomness.
            chunk_size (int, optional): The size of text chunks to split documents into. Defaults to 3500.
            chunk_overlap (int, optional): The overlap between consecutive text chunks. Defaults to 0.
        """
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = ChatGroq(api_key=groq_key, model=self.model, temperature=self.temperature)  # Initialize the language model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
        self.system_logger = system_logger  # Logger for system-level events
        self.userops_logger = userops_logger  # Logger for user operations
        self.llm_logger = llmresponse_logger  # Logger for LLM responses

    def create_contextual_chunks(self, file_path: str):
        """
        Splits a document into contextual chunks and adds metadata and context to each chunk.

        Args:
            file_path (str): Path to the document file.

        Returns:
            list: A list of Document objects with added context and metadata.

        Raises:
            Exception: If an error occurs during chunk creation.
        """
        self.system_logger.info(f"Creating contextual chunks for {file_path}")
        try:
            # Load the document using PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            doc_pages = loader.load()

            # Split the document into chunks using RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            doc_chunks = splitter.split_documents(doc_pages)
            original_doc = '\n'.join([doc.page_content for doc in doc_chunks])  # Combine all chunks into a single string

            # Add context and metadata to each chunk
            contextual_chunks = []
            for chunk in doc_chunks:
                chunk_content = chunk.page_content
                chunk_metadata = chunk.metadata
                chunk_metadata_upd = {
                    "id": str(uuid.uuid4()),  # Generate a unique ID for the chunk
                    "page": chunk_metadata["page"],  # Page number from the original document
                    "source": chunk_metadata["source"],  # Source file path
                    "title": chunk_metadata["source"].split("/")[-1]  # Extract the file name as the title
                }
                context = generate_chunk_context(original_doc, chunk_content)  # Generate context for the chunk
                contextual_chunks.append(Document(page_content=context + '\n' + chunk_content,
                                                 metadata=chunk_metadata_upd))
            self.system_logger.info(f"Created {len(contextual_chunks)} chunks")
            return contextual_chunks

        except Exception as e:
            self.system_logger.error(f"Error creating contextual chunks: {str(e)}")
            raise

    @property
    def get_embeddings(self):
        """
        Initializes and returns the embedding model for text vectorization.

        Returns:
            HuggingFaceEmbeddings: The initialized embedding model.

        Raises:
            Exception: If embedding model initialization fails.
        """
        try:
            self.system_logger.info("Initializing embedding model")
            model_name = "BAAI/bge-small-en-v1.5"  # Name of the embedding model
            model_kwargs = {'device': self.device}  # Specify the device (CPU/GPU)
            encode_kwargs = {'normalize_embeddings': True}  # Normalize embeddings for consistency

            # Initialize the HuggingFace embeddings model
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
        """
        Initializes and returns a document retriever based on the specified method.

        Args:
            file_path (str): Path to the document file.
            method (str, optional): Retrieval method to use. Options: "similarity", "bm25", "hybrid", "rerank". Defaults to "hybrid".

        Returns:
            any: A retriever object based on the specified method.

        Raises:
            Exception: If an error occurs during retriever setup.
        """
        self.system_logger.info(f"Setting up {method} retriever")
        try:
            chunks = self.create_contextual_chunks(file_path)  # Create contextual chunks from the document

            if method == "similarity":
                # Use Chroma for similarity-based retrieval
                retriever = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.get_embeddings,
                    collection_name="similarity_collection",
                    collection_metadata={"hnsw:space": "cosine"}
                ).as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 similar documents

            elif method == "bm25":
                # Use BM25 for keyword-based retrieval
                retriever = BM25Retriever.from_documents(documents=chunks, k=5)

            elif method == "hybrid":
                # Combine similarity and BM25 retrievers for hybrid retrieval
                similarity_retriever = self.get_retriever(file_path, "similarity")
                bm25_retriever = self.get_retriever(file_path, "bm25")

                retriever = EnsembleRetriever(
                    retrievers=[similarity_retriever, bm25_retriever],
                    weights=[0.5, 0.5]  # Equal weights for both retrievers
                )

            elif method == "rerank":
                # Use a reranker to refine retrieval results
                reranker = CrossEncoderReranker(
                    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"),
                    top_n=5  # Rerank top 5 documents
                )

                retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=self.get_retriever(file_path, "hybrid")  # Use hybrid retriever as base
                )

            else:
                raise ValueError(f"Unknown retrieval method: {method}")

            self.system_logger.info(f"{method} retriever setup complete")
            return retriever

        except Exception as e:
            self.system_logger.error(f"Error setting up {method} retriever: {str(e)}")
            raise

    def create_qa_chain(self, file_path: str, retrieval_method: str = "rerank"):
        """
        Creates a question-answering (QA) chain using the specified retrieval method.

        Args:
            file_path (str): Path to the document file.
            retrieval_method (str, optional): Retrieval method to use. Defaults to "rerank".

        Returns:
            any: A QA chain object for answering questions.

        Raises:
            Exception: If an error occurs during QA chain creation.
        """
        self.system_logger.info("Creating QA chain")
        try:
            retriever = self.get_retriever(file_path, retrieval_method)  # Initialize the retriever

            # Define the prompt template for the QA task
            prompt = """You are an expert in question-answering tasks.
                        Answer the following question using only the provided context.
                        If the answer cannot be found in the context, state that you don't know.
                        Provide detailed, well-formatted answers based on the context.

                        Question: {question}

                        Context: {context}

                        Answer:"""

            prompt_template = ChatPromptTemplate.from_template(prompt)

            # Create the QA chain
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
        """
        Generates a response to a user question using the QA chain.

        Args:
            qa_chain: The QA chain object.
            question (str): The user's question.

        Returns:
            str: The generated response.

        Raises:
            Exception: If an error occurs during response generation.
        """
        self.userops_logger.info(f"Processing query: {question}")
        try:
            response = qa_chain.invoke(question)  # Invoke the QA chain to generate a response
            self.llm_logger.info(f"Question: {question}\nResponse: {response}")
            return response
        except Exception as e:
            self.system_logger.error(f"Error processing query: {str(e)}")
            raise

   