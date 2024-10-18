<h1 align="center"><b>Generative AI Roadmap</b></h1>

Generative AI is a subset of Artificial Intelligence where content we required is **Generated** with the help of **Prompts.**

**prompts:** Prompts are the piece of text where we need to give to AI, To generate more content.

### **Types of prompts:**
- **System Prompt:** System prompts are used to add text in the programming languages, where it won't change again and again. We can insert User prompt into system prompt in some cases.
- **User Prompt:** User prompts are used to add text from the user to AI.

## Before stepping into Generative AI , we have to make sure to learn more about working of Large Language Models (LLMs).
for that we have to learn

- Mathematics Behind Machine Learning
    - Linear Algebra
    - multivariate calculus
    - statistics and probability
- Machine Learning
- Deep Learning
    - check out [ML_Libraries.md](ML_LIBRARIES.md) to learn more on machine learning and deep learning libraries in python.
- Working of Transformer model. Read "Attention is all you need" research paper to know more about transformers.

### _**Python libraries to learn Generative AI**_

  #### **1. Langchain 🦜 [[Link]](https://python.langchain.com/v0.2/docs/introduction/)**
  
  - LangChain is an open-source framework designed to simplify the development of applications that utilize large language models (LLMs).
  - LangChain enables developers to combine powerful LLMs, such as those from OpenAI (e.g., GPT-3.5, GPT-4), with external data sources and other components to build robust NLP applications.
  - It provides a standardized interface for interacting with various LLMs and integrating them with different data sources and software workflows.
  
  #### **2. LangSmith 🦜🛠️ [[Link]](https://docs.smith.langchain.com/)**
  
  - LangSmith is a comprehensive platform designed to support the development, deployment, and maintenance of Large Language Model (LLM) applications, particularly focusing on the transition from prototyping to production.

### _**Large Language Models (LLMs)**_

- A Large Language Model (LLM) is a sophisticated type of artificial intelligence (AI) designed to process, understand, and generate human language.
- LLMs are deep learning algorithms that use massive datasets to recognize, generate, translate, and summarize vast quantities of written human language. They are a subset of machine learning, specifically within the domain of natural language processing (NLP).

### _**Various types of Large Language Models**_

  ### 1. For text generation 🗒️

  - **ChatGPT: [[Link]](http://chat.openai.com/)** ChatGPT is based on the Generative Pre-trained Transformer (GPT) architecture, specifically using the GPT-3.5 and later updated to GPT-4 models. These models are trained on massive datasets, approximately 45 terabytes of text from the internet, to learn patterns and associations in language.
  - **Google Gemini: [[Link]](https://gemini.google.com/)** Google Gemini is designed to be multimodal, meaning it can seamlessly understand, process, and generate various types of data, including text, code, images, audio, and video. This multimodality allows Gemini to perform a wide range of tasks across different domains.
  - **Claude: [[Link]](https://claude.ai/)** Claude is trained to engage in natural, text-based conversations and excels in tasks such as summarization, editing, Q&A, decision-making, and code writing. As of Claude 3, the model can also analyze and transcribe static images, including handwritten notes and photographs.
  - **Llama: [[Link]](https://www.llama.com/docs/get-started/)** Llama, which stands for Large Language Model Meta AI, is a family of autoregressive large language models developed by Meta AI. Which is free and open source LLM and can run locally.
  - **Mistral: [[Link]](https://docs.mistral.ai/)** Mistral AI is a French company specializing in the development and deployment of advanced artificial intelligence, particularly large language models. his model uses a sparse mixture of experts architecture, with 46.7 billion parameters but only using 12.9 billion parameters per token. It beats benchmarks set by LLaMA 70B and GPT-3.5 in many tests.

  ### 2. For image generation 🖼️
  
 - **Flux.1: [[Link]](https://blackforestlabs.ai/announcements/)** The FLUX.1 model is a state-of-the-art text-to-image AI generator developed by Black Forest Labs,  based on a hybrid architecture of multimodal and parallel diffusion transformer blocks, scaled to 12 billion parameters.
 - **Stable Diffusion (SDXL): [[Link]](https://platform.stability.ai/docs/)** Stable Diffusion XL (SDXL) is a advanced text-to-image generation model developed by Stability AI, representing a significant improvement over previous versions of the Stable Diffusion models.
 - **Midjourney: [[Link]](https://www.midjourney.com/home)** Midjourney is an AI-powered image generation platform that transforms text prompts into visually striking images. It operates primarily through the Discord platform. Users need to create or use an existing Discord account to interact with the Midjourney bot, which generates images based on user inputs.
 - **DALL-E: [[Link]](https://openai.com/index/dall-e-3/)** DALL-E is a series of generative AI models developed by OpenAI, designed to create images from textual descriptions. DALL-E stands for "Decoder-Only Autoregressive Language and Image Synthesis" and is named after a combination of Salvador Dalí and the Pixar character WALL-E. It uses deep learning techniques, specifically transformer architectures, to generate images based on text prompts.

### Retrieval Augumented Generation (RAG)

- Retrieval-Augmented Generation (RAG) is an AI framework designed to enhance the accuracy, reliability, and relevance of responses generated by large language models (LLMs).
- RAG integrates the capabilities of traditional information retrieval systems with the generative powers of LLMs. This approach involves retrieving relevant information from an external knowledge base or database and using this information to augment the input to the LLM, thereby improving the accuracy and contextuality of the generated text.


<p align="center"><img src="https://github.com/user-attachments/assets/68b4957b-5a82-43b6-aef8-7b3f86340ad3" /></p>


### Embedding models

- Embedding models are a fundamental concept in machine learning and natural language processing, designed to represent complex data such as text, images, and audio as vectors in a high-dimensional space.
- Embeddings are representations of objects (like words, images, or audio) as vectors in a continuous vector space. This transformation allows machine learning models to understand and process these objects in a way that captures their semantic relationships and similarities.

### Check out [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to select top embedding models for your RAG application up-to-date.

## Vector databases

- Vector databases are specialized systems designed to efficiently manage, store, and query high-dimensional vector data, which is particularly useful in the context of artificial intelligence, machine learning, and modern search engines.

### Examples of Vector Databases

### 1. Weaviate [[Link]](https://weaviate.io/developers/weaviate)
- Weaviate is an open-source vector database that allows users to store data objects and their corresponding vector embeddings. It supports scalability into billions of data objects and uses GraphQL API for querying. Weaviate is particularly useful for applications like recommendation systems, semantic search, and enhancing large language models (LLMs) with long-term memory.

### 2. Pinecone [[Link]](https://docs.pinecone.io/home)
- Pinecone is a managed, cloud-native vector database that offers a straightforward API and does not require infrastructure maintenance. It is known for its high-speed data processing, metadata filters, and sparse-dense index support, making it suitable for real-time applications and large-scale data handling.

### 3. Milvus [[Link]](https://milvus.io/docs)
- Milvus is an open-source vector database designed for AI and analytics workloads. It supports similarity search at scale and heterogeneous computing, making it well-suited for machine learning applications such as semantic search and recommendation systems. Milvus is scalable and can handle large datasets efficiently.

### 4. Faiss [[Link]](https://faiss.ai/index.html)
- Faiss (Facebook AI Similarity Search) is an open-source library for efficient similarity search and clustering of dense vectors. It is primarily coded in C++ but supports Python/NumPy integration and can execute some algorithms on GPUs. Faiss is useful for searching within large vector sets that may exceed RAM capacity.

### 5. Qdrant [[Link]](https://qdrant.tech/documentation/)
- Qdrant is a vector database and API service that enables searches for the closest high-dimensional vectors. It is used for tasks like matching, searching, making recommendations, and more. Qdrant supports various indexing algorithms and is designed for real-time vector similarity searches.

### 6. Chroma [[Link]](https://docs.trychroma.com/getting-started)
- Chroma is an open-source embedding database that simplifies building LLM applications by making knowledge, facts, and skills pluggable for LLMs. It supports queries, filtering, density estimates, and integrates with frameworks like LangChain. Chroma is scalable from Python notebooks to production clusters.

### 7. MongoDB [[Link]](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- MongoDB, through its Atlas Vector Search, can handle vector data alongside traditional transactional data. It uses a specialized vector index that is automatically synced with the core database, allowing for efficient vector searches without compromising on the flexibility of MongoDB's document-based data model.

### 8. Deep Lake [[Link]](https://www.deeplake.ai/)
- Deep Lake is another vector database that integrates well with deep learning workflows. It is designed to handle large-scale vector data and supports various indexing algorithms, making it suitable for applications requiring fast and accurate similarity searches.

### 9. Redis [[Link]](https://redis.io/docs/latest/develop/interact/search-and-query/)
- **Redis with RediSearch**: This combination allows for efficient vector searches within Redis, leveraging its in-memory capabilities for high performance.

### 10. Pgvector [[Link]](https://github.com/pgvector/pgvector)
- pgvector is an open-source extension for PostgreSQL that enables the storage, querying, and indexing of vector data, particularly useful in the context of machine learning and AI applications. 

## Graph Databases
- Graph databases are specialized databases designed to store and manage data as a network of nodes and edges, rather than in the traditional table-based or document-based structures of relational or NoSQL databases.

### Examples of graph databases

Here are some examples of graph databases, each with their unique features and use cases:

### Neo4j [[Link]](https://neo4j.com/docs/)
- A native property graph database, known for its high performance and scalability. It is widely used in various applications, including social networks, recommendation systems, and master data management. Neo4j supports the Cypher query language and has both community and enterprise versions.

### Amazon Neptune [[Link]](https://docs.aws.amazon.com/neptune/)
- A fully managed graph database service provided by Amazon Web Services (AWS). It supports both property graphs and RDF graphs, and uses query languages such as Gremlin, SPARQL, and openCypher. Neptune is designed for superior scalability and availability, making it suitable for cloud-based applications.

### ArangoDB [[Link]](https://docs.arangodb.com/stable/)
- A multi-model database that supports property graphs, documents, and key-value stores. It uses the ArangoDB Query Language (AQL) and is known for its flexibility and scalability. ArangoDB offers both open-source and commercial versions, making it a versatile option for various use cases.

### JanusGraph [[Link]](https://docs.janusgraph.org/)
- An open-source, distributed graph database optimized for storing and querying large graphs. It supports various storage backends like Apache Cassandra, Apache HBase, and Google Cloud Bigtable. JanusGraph is integrated with big data platforms such as Apache Spark and Apache Hadoop, and supports global graph data analytics.

### TigerGraph [[Link]](https://docs.tigergraph.com/home/)
- A native parallel graph database designed for enterprise-scale applications. It is known for its speed and ability to handle massive amounts of data and deep relationship queries in real-time. TigerGraph offers both commercial and freemium options.

### Azure Cosmos DB [[Link]](https://learn.microsoft.com/en-us/azure/cosmos-db/)
- A multi-model database service provided by Microsoft Azure, which includes a graph database component using the Apache Gremlin query language. It is designed to store massive graphs with billions of vertices and edges and offers millisecond latency for queries.

### OrientDB [[Link]](https://orientdb.org/docs/3.1.x/)
- A second-generation distributed graph database that also supports document-oriented data storage. It offers full ACID support, multi-master replication, and a query language similar to SQL. OrientDB is licensed under the Apache 2 license for the community edition.

### RedisGraph [[Link]](https://redis.io/docs/latest/operate/oss_and_stack/stack-with-enterprise/deprecated-features/graph/)
- An in-memory, queryable property graph database that uses sparse matrices and linear algebra for querying. It is part of the Redis ecosystem and is known for its high performance and ease of use.

### SAP HANA Graph [[Link]](https://help.sap.com/doc/21574acf46fe45a8ae9def213f2c4d9e/2.0.05/en-US/SAP_HANA_Graph_Reference_en.pdf)
- An in-memory property graph database integrated into the SAP HANA platform. It supports ACID transactions and is designed for high-performance querying of large networks.

### Dgraph [[Link]](https://dgraph.io/docs/)
- A distributed graph database that uses GraphQL as its query language. It is known for its performance and scalability, making it suitable for large-scale applications. Dgraph is highly regarded for its ability to handle billions of edges and vertices.

### Cayley [[Link]](https://cayley.gitbook.io/cayley)
- An open-source graph database written in Go, inspired by the graph database behind Freebase and Google's Knowledge Graph. It is designed for simplicity and scalability, making it a good option for developers looking for a lightweight solution.

## LLM inferencing APIs

### 1. Groq [[Link]](https://groq.com/)

Groq is an AI accelerator designed to enhance the performance of large-scale natural language processing (NLP) tasks. It is specialized for fast and efficient language model processing.

### 2. Cerebras [[Link]](https://cerebras.ai/)

Develops a new class of computers designed to train AI models efficiently, with applications in NLP, computer vision, and more.


## AI Agents

AI agents built on large language models (LLMs) represent a significant advancement in the field of artificial intelligence, combining the capabilities of LLMs with additional components to handle complex tasks more effectively.

### Tools to create AI agents

To create AI agents, you can utilize a variety of tools and frameworks, each offering different levels of customization, ease of use, and specific functionalities. Here are some of the notable tools and frameworks:

## Frameworks and Platforms

### Mosaic AI Agent Framework [[Link]](https://www.databricks.com/product/machine-learning/retrieval-augmented-generation)
- This framework, as described by Databricks, allows you to create AI agents using the AI Playground. You can define tools as Unity Catalog functions or local Python functions and integrate them with LLMs. The framework supports exporting and deploying agents to Model Serving endpoints.

### LangChain [[Link]](https://python.langchain.com/v0.2/docs/introduction/)
- An open-source framework that provides a modular architecture for building AI agents. It is highly customizable and can be used to create agents that interact with various tools and services.

### LangGraph [[Link]](https://langchain-ai.github.io/langgraph/)
- LangGraph is a library within the LangChain ecosystem designed to facilitate the creation, coordination, and execution of complex agent workflows, particularly those involving large language models (LLMs).

### Microsoft Autogen [[Link]](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/)
- Known for its easy collaboration features and simplified agent building process. This framework is useful for businesses looking for a more streamlined approach to creating AI agents.

### LlamaIndex [[Link]](https://docs.llamaindex.ai/en/stable/)
- Specializes in tasks related to information retrieval and can be integrated into AI agent frameworks to enhance their capabilities.

### AgentGPT [[Link]](https://docs.reworkd.ai/introduction)
- A web-based platform supported by an open-source community, allowing users to build and deploy AI agents directly from their browsers. It is designed for autonomy and customization.

### AutoGPT [[Link]](https://docs.agpt.co/)
- An experimental, open-source autonomous AI agent built on the GPT-4 language model. It links together tasks to accomplish user-defined goals and offers high customization and flexibility.

### Zapier Central [[Link]](https://zapier.com/central)
- A no-code AI agent builder that integrates with Zapier’s extensive catalog of apps, enabling easy creation of AI agents without coding knowledge.

### CrewAI [[Link]](https://docs.crewai.com/introduction)
- A paid builder platform with pre-built components and tools for building AI assistants. It is known for its ease of use and advanced LLM integration.


## PDF scrapping resources to build RAG

For extracting data from PDF files using Python, there are several open-source libraries that are highly recommended. Here are some of the most effective ones:

### PyMuPDF (fitz) [[Link]](https://pymupdf.readthedocs.io/en/latest/)
- PyMuPDF, often imported as `fitz`, is widely regarded as one of the best open-source libraries for extracting text from PDFs. It provides a structured extract of the PDF content and is particularly good at handling text that is visibly seen when viewing the PDF. You can use methods like `page.get_text('blocks')` or `page.get_text('dict')` to extract text in a structured format.

### PDFMiner [[Link]](https://readthedocs.org/projects/pdfminer-docs/downloads/pdf/latest/)
- PDFMiner is a user-friendly and open-source library that focuses primarily on extracting text from PDF documents. It offers better accuracy and customization options compared to some other libraries. It works offline and is readily available for implementation.

### PDFPlumber [[Link]](https://python.langchain.com/docs/integrations/document_loaders/pdfplumber/)
- PDFPlumber is built on top of PDFMiner.Six and offers a simpler interface for extracting text and metadata from PDFs. It provides detailed information about each text character, rectangle, and line, and also supports table extraction and visual debugging.

### PyPDF2 [[Link]](https://pypdf2.readthedocs.io/en/3.0.0/index.html)
- While PyPDF2 is a pure-python library that can handle various PDF tasks, including text extraction, it has been noted to be less reliable for text extraction compared to PyMuPDF and PDFPlumber. However, it is still useful for tasks like merging, splitting, and encrypting PDF files.

### Tabula-py [[Link]](https://tabula-py.readthedocs.io/en/latest/)
- Tabula-py is specifically designed for extracting tables from PDF documents and converting them into pandas DataFrames or CSV/JSON files. It is very useful for tabular data extraction.

### pdfminer.six [[Link]](https://pdfminersix.readthedocs.io/en/latest/)
- This is an updated version of the original PDFMiner library and is used for extracting text from PDF documents. It is known for its simplicity and effectiveness.


## Pdf scrapping services to use in RAG

### 1.Unstructured.io [[Link]](https://unstructured.io/)
- Unstructured.io is a platform and a set of tools designed to preprocess and transform unstructured data into a format that can be easily integrated into machine learning pipelines, particularly for Large Language Models (LLMs).

### 2.LlamaParse [[Link]](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/)
- LlamaParse is a proprietary document parsing service developed by LlamaIndex, designed to enhance the capabilities of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) applications.
  
### 3.LLMWhisperer [[Link]](https://unstract.com/llmwhisperer/)
- This is the most common and practical application of LLMWhisperer. It is a technology designed to preprocess and present data from complex documents in a format that Large Language Models (LLMs) can effectively understand.
