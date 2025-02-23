# Chat-based Contextual RAG Implementation

## Overview
This repository contains an implementation of a **Contextual RAG (Retrieval-Augmented Generation)** system for document-based question-answering. Contextual RAG is a concept proposed by **Anthropic** to improve the quality of retrieval-augmented results. This method enhances standard semantic search by incorporating **hybrid search** (BM25 + dense embeddings) and **reranking** to refine the retrieved results before passing them to the language model for response generation.

## Features
- **Contextual Chunking:** Documents are split into chunks with added context before embedding generation.
- **Multiple Retrieval Methods:** Supports similarity search, BM25, hybrid retrieval, and reranking.
- **QA Chain:** Combines retrieval and LLM inference to generate detailed answers.
- **Hybrid Search:** Combines BM25 keyword-based search with dense vector similarity search.
- **Reranking:** Uses cross-encoder-based reranking for improved result quality.
- **Logging:** System logs to track different operations, including user queries, retrieval steps, and LLM responses.

## How Contextual RAG Works
The method aims to improve standard RAG by adding **explanatory context to each chunk**. Instead of retrieving raw text chunks, it prepends chunk-specific explanatory context based on the entire document. Then, embeddings and TF-IDF vectors are generated for improved retrieval. The goal is to provide a **more coherent and contextually aware retrieval** before the query is answered.

## Challenges & Limitations
While the approach is theoretically strong, practical implementation comes with significant **challenges**:

- **Costly to Run:** The approach **requires extensive embedding generation and TF-IDF processing**, which can be expensive if using **OpenAI embeddings and models**. Unfortunately, as a **poor man**, I can't afford OpenAI’s high API costs, making this method impractical for large-scale deployment.
- **Inefficiency with Large Documents:** The **chunk-specific context generation process becomes infeasible for large documents**, as it requires substantial compute resources and time.
- **Model Limitations:** Groq models, which I had access to, have a **TPM (Tokens Per Minute) limit of 6000**, making this method **unusable for large documents**. Only **small documents (10-15 pages max)** can realistically work within these constraints.
- **Slow Execution:** The retrieval and reranking process, combined with the **added context generation**, significantly slows down the response time. However, this might be an **issue on my side**, and further experimentation is needed to optimize performance.

## Future Improvements
- Exploring **alternative embedding models** with better cost efficiency.
- Testing **different retrieval strategies** to balance cost and quality.
- Optimizing **processing pipelines** for speed improvements.
- Investigating **better hardware setups** to reduce bottlenecks in retrieval and reranking.

## Conclusion
Contextual RAG is a promising technique for **improving retrieval-augmented question answering**, but it comes with significant **cost and efficiency challenges**. While the approach is theoretically superior to standard semantic search, real-world constraints (API costs, model limitations, and processing time) make it **difficult to scale** without further optimizations.

