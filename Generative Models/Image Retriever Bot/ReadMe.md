# Image Retrieval Bot - README

### Overview
Image Retrieval Bot allows users to search and retrieve relevant images based on text queries. It uses CLIP for image-text embeddings, FAISS for efficient similarity search, and Flan-T5 for query processing. The bot can be extended to handle PDFs, videos, and larger datasets.
Note: Using a better LLM like llama 3.2 or openAI will generate a much more efficent response.

### Features
- **CLIP Embeddings**: Maps images and text into a common space for comparison.
- **FAISS Search**: Enables fast similarity-based image retrieval.
- **Query-Based Retrieval**: Finds images matching text descriptions.
- **Text Generation**: Utilizes Flan-T5 for text-based queries and responses.
- **Scalable**: Easily adaptable to larger datasets and multimedia.

Extending
- **PDF/Video Search**: Adapt document processing for cross-media retrieval.
