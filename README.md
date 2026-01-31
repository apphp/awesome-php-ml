# Awesome PHP Machine Learning & AI

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub stars](https://img.shields.io/github/stars/apphp/awesome-php-ml?style=social)](https://github.com/apphp/awesome-php-ml)
[![Last commit](https://img.shields.io/github/last-commit/apphp/awesome-php-ml)](https://github.com/apphp/awesome-php-ml/commits)
[![License](https://img.shields.io/github/license/apphp/awesome-php-ml)](https://github.com/apphp/awesome-php-ml/blob/main/LICENSE)
[![Link Check](https://github.com/apphp/awesome-php-ml/actions/workflows/link-check.yml/badge.svg)](https://github.com/apphp/awesome-php-ml/actions/workflows/link-check.yml)

The most comprehensive curated list of **Machine Learning, Artificial Intelligence, NLP, LLM and Data Science libraries for PHP**.

Inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and the broader **Awesome** ecosystem.

> **Goal:** make it easy to build intelligent systems with PHP — from classic ML to modern LLM-based workflows.

## What is this?

- Curated list of **PHP libraries and tools** for Machine Learning, AI, NLP, LLMs and Data Science.
- Focused on **code-first resources**: packages, SDKs, frameworks, and building blocks.
- Aimed at **PHP developers** who want to add intelligent features to existing apps or build new AI-powered systems.

## How to use this list

- **Classic ML / traditional models** – start with [php-ai/php-ml](https://gitlab.com/php-ai/php-ml) and [RubixML/RubixML](https://github.com/RubixML/RubixML).
- **LLM-powered apps & agents** – see [LLMs & AI APIs](#llms--ai-apis), [Embeddings & Vector Search](#embeddings--vector-search), and framework integrations (Laravel/Symfony).
- **RAG (Retrieval-Augmented Generation)** – combine [php-rag](https://github.com/mzarnecki/php-rag) with vector databases like [pgvector](https://github.com/pgvector/pgvector) or [Meilisearch](https://github.com/meilisearch/meilisearch-php).
- **Numerical computing & math** – explore [Math, Statistics & Linear Algebra](#math-statistics--linear-algebra) for tensors, matrices, and statistics.
- **Production integration** – use [Interop & Model Serving](#interop--model-serving) and framework integrations to wire models into real apps.

### Example "recipes"

- **I want to build a Laravel RAG app**  
  Use an LLM client like 🌟 [openai-php/client](https://github.com/openai-php/client), embeddings + vector search via 🌟 [LLPhant/LLPhant](https://github.com/LLPhant/LLPhant) with 🌟 [pgvector/pgvector](https://github.com/pgvector/pgvector) or 🌟 [meilisearch/meilisearch-php](https://github.com/meilisearch/meilisearch-php), and orchestrate agents/RAG flows with 🌟 [neuron-core/neuron-ai](https://github.com/neuron-core/neuron-ai), integrating into Laravel using 🌟 [openai-php/laravel](https://github.com/openai-php/laravel) and the packages under [Laravel & Framework Integrations](#laravel--framework-integrations).

- **I only need translation or vision**  
  For translation, see 🌟 [deepl-php](https://github.com/DeepLcom/deepl-php) and 🌟 [googleapis/google-cloud-php](https://github.com/googleapis/google-cloud-php) under [Interop & Model Serving](#interop--model-serving). For image/vision workloads, combine [Computer Vision, Image & Video Processing](#computer-vision-image--video-processing) libraries with cloud AI services via 🌟 [symfony/ai](https://github.com/symfony/ai) or [openai-php/client](https://github.com/openai-php/client) from [LLMs & AI APIs](#llms--ai-apis).

### Recommended core stack

These are opinionated defaults you can reach for when you just want something that works in production.

- **General ML:** 🌟 [RubixML/RubixML](https://github.com/RubixML/RubixML) for end-to-end ML pipelines.
- **LLM clients:** 🌟 [openai-php/client](https://github.com/openai-php/client) and 🌟 [google-gemini-php/client](https://github.com/google-gemini-php/client) for major model providers.
- **Embeddings & vector search:** 🌟 [LLPhant/LLPhant](https://github.com/LLPhant/LLPhant) with 🌟 [pgvector/pgvector](https://github.com/pgvector/pgvector), 🌟 [pgvector/pgvector-php](https://github.com/pgvector/pgvector-php), 🌟 [meilisearch/meilisearch-php](https://github.com/meilisearch/meilisearch-php) or 🌟 [algolia/algoliasearch-client-php](https://github.com/algolia/algoliasearch-client-php).
- **Data processing:** 🌟 [flow-php/flow](https://github.com/flow-php/flow) for typed ETL-style pipelines.
- **Interop with Python ML:** 🌟 [swoole/phpy](https://github.com/swoole/phpy) to call into the Python ecosystem when needed.

## Legend

Not all projects are tagged yet – we’re gradually adding markers as the ecosystem evolves. Treat them as rough guidance, not strict rules.

- `🌟` – widely used / production-ready projects
- `🧪` – experimental or research-oriented projects
- `⚠️` – projects with limited maintenance, older APIs, or niche usage; review before using in new projects.

---

## Contents

- [What is this?](#what-is-this)
- [How to use this list](#how-to-use-this-list)
- [Recommended core stack](#recommended-core-stack-)
- [Legend](#legend)
- [Machine Learning](#machine-learning)
- [Deep Learning & Neural Networks](#deep-learning--neural-networks)
- [Natural Language Processing](#natural-language-processing)
- [Computer Vision, Image & Video Processing](#computer-vision-image--video-processing)
- [Math, Statistics & Linear Algebra](#math-statistics--linear-algebra)
- [LLMs & AI APIs](#llms--ai-apis)
- [Embeddings & Vector Search](#embeddings--vector-search)
- [Data Processing](#data-processing)
- [Interop & Model Serving](#interop--model-serving)
- [Tools & Utilities](#tools--utilities)
- [Laravel & Framework Integrations](#laravel--framework-integrations)
- [Symfony & Framework Integrations](#symfony--framework-integrations)
- [Resources](#resources)

---

## Machine Learning

*Core PHP libraries for supervised/unsupervised learning, classification, regression, and clustering.*

- [dr-que/polynomial-regression](https://github.com/jbboehr/PolynomialRegression.php) – Polynomial regression for PHP
- [kyriansfriends/transformers](https://github.com/kyriansfriends/transformers) — A PHP toolkit for running Hugging Face–style Transformer models with ONNX Runtime (text generation, summarization, classification, etc.)
- [php-ai/php-ml](https://gitlab.com/php-ai/php-ml) – Core machine learning algorithms for PHP
- [php-ai/php-ml-examples](https://github.com/php-ai/php-ml-examples) – Practical examples for PHP-ML

---

## Deep Learning & Neural Networks

*PHP libraries for neural networks, deep learning architectures, and advanced learners built on tensors.*

- 🌟 [RubixML/RubixML](https://github.com/RubixML/RubixML) – ![GitHub stars](https://img.shields.io/github/stars/RubixML/RubixML?style=social) High-level ML framework with pipelines and datasets

---

## Natural Language Processing

*Text processing, tokenization, language detection, sentiment analysis and other NLP tasks in PHP.*

- ⚠️ [angeloskath/php-nlp-tools](https://github.com/angeloskath/php-nlp-tools) – Natural Language Processing tools
- [CodeWithKyrian/transformers-php](https://github.com/CodeWithKyrian/transformers-php) – Hugging Face–style Transformer inference in PHP using ONNX
- [friteuseb/nlp_tools](https://github.com/friteuseb/nlp_tools) – Extension for NLP methods and text analysis
- ⚠️ [patrickschur/language-detection](https://github.com/patrickschur/language-detection) – Language detection library
- 🧪 [RubixML/Sentiment](https://github.com/RubixML/Sentiment) — Example project demonstrating sentiment analysis with a neural network (IMDB reviews) using Rubix ML in PHP
- [voku/stop-words](https://github.com/voku/stop-words) – Stop word lists for many languages
- [yooper/php-text-analysis](https://github.com/yooper/php-text-analysis) – Sentiment analysis and NLP tools

---

## Computer Vision, Image & Video Processing

*Image manipulation, preprocessing, and computer vision workloads from PHP.*

- [aschmelyun/subvert](https://github.com/aschmelyun/subvert) - Generate subtitles, summaries, and chapters from videos in seconds
- [Intervention/image](https://github.com/Intervention/image) – Image manipulation library for CV preprocessing
- [jcupitt/vips](https://github.com/jcupitt/libvips) – Fast image processing library with PHP bindings
- [php-opencv/php-opencv](https://github.com/php-opencv/php-opencv) – OpenCV bindings for PHP

---

## Math, Statistics & Linear Algebra

*Numerical computing, matrix operations, statistics, and related math foundations for ML and data science in PHP.*

- 🌟 [brick/math](https://github.com/brick/math) — Arbitrary-precision arithmetic for PHP (BigInteger, BigDecimal, BigRational)
- [markrogoyski/math-php](https://github.com/markrogoyski/math-php) – Math library for linear algebra, statistics, and calculus
- [mcordingley/LinearAlgebra](https://github.com/mcordingley/LinearAlgebra) – Stand-alone linear algebra library
- ⚠️ [NumPHP/NumPHP](https://github.com/NumPHP/NumPHP) – Math library for scientific computing
- [rindow/rindow-math-matrix](https://github.com/rindow/rindow-math-matrix) – Foundational package for scientific matrix operations
- [RubixML/numpower](https://github.com/RubixML/numpower) – High-performance numerical computing library inspired by NumPy
- 🌟 [RubixML/Tensor](https://github.com/RubixML/Tensor) – ![GitHub stars](https://img.shields.io/github/stars/RubixML/Tensor?style=social) N-dimensional tensors for numerical computing

---

## LLMs & AI APIs

*Clients, SDKs, and frameworks for calling hosted LLMs and other AI providers from PHP.*

- [adrienbrault/instructrice](https://github.com/adrienbrault/instructrice) — Typed LLM outputs in PHP with flexible schema support (OpenAI, Claude, Gemini, etc.) and type-safe handling of structured responses
- [ArdaGnsrn/ollama-php](https://github.com/ArdaGnsrn/ollama-php) – A PHP client library for the Ollama LLM server, enabling completions, chat, model management, and embeddings via Ollama’s API
- [cognesy/instructor-php](https://github.com/cognesy/instructor-php) – Structured-output helper for LLM responses
- [deepseek-php/deepseek-php-client](https://github.com/deepseek-php/deepseek-php-client) – PHP client library for integrating with the DeepSeek AI API, providing a fluent API for model queries, streaming results, and support for multiple HTTP clients and models
- [elastic/elasticsearch-chatgpt-php](https://github.com/elastic/elasticsearch-chatgpt-php) – Experimental PHP library that uses ChatGPT to translate natural language into Elasticsearch DSL queries and perform semantic search over your indices
- [FunkyOz/mulagent](https://github.com/FunkyOz/mulagent) – Multi-agent orchestration framework for LLM applications
- ⚠️ [HosonoDE/EasyAI-PHP](https://github.com/HosonoDE/EasyAI-PHP) – High-level AI integration library for PHP that simplifies using LLMs
- 🌟 [google-gemini-php/client](https://github.com/google-gemini-php/client) – ![GitHub stars](https://img.shields.io/github/stars/google-gemini-php/client?style=social) Gemini PHP is a community-maintained PHP API client that allows you to interact with the Gemini AI API
- 🌟 [kambo-1st/langchain-php](https://github.com/kambo-1st/langchain-php) ![GitHub stars](https://img.shields.io/github/stars/kambo-1st/langchain-php?style=social) A PHP port of the LangChain framework for building composable LLM-powered applications
- 🌟 [llm-agents-php/agents](https://github.com/llm-agents-php/agents) - ![GitHub stars](https://img.shields.io/github/stars/llm-agents-php/agents?style=social) LM Agents is a PHP library for building and managing Language Model (LLM) based agents
- [llm-agents-php/prompt-generator](https://github.com/llm-agents-php/prompt-generator) - Prompt generator for LLM agents with interceptors
- [ModelFlow-AI (GitHub org)](https://github.com/modelflow-ai) – Collection of PHP packages for unified access to AI models, embeddings, and chat (OpenAI, Mistral, Ollama) 
- [mzarnecki/php-rag](https://github.com/mzarnecki/php-rag) – PHP RAG toolkit for connecting vector search and LLMs in retrieval-augmented workflows
- 🌟 [openai-php/client](https://github.com/openai-php/client) – ![GitHub stars](https://img.shields.io/github/stars/openai-php/client?style=social) Official OpenAI PHP client
- 🌟 [orhanerday/open-ai](https://github.com/orhanerday/open-ai) – ![GitHub stars](https://img.shields.io/github/stars/orhanerday/open-ai?style=social) Popular OpenAI PHP SDK
- [sarfraznawaz2005/ai-team](https://github.com/sarfraznawaz2005/ai-team) — Package to build and run collaborative teams of AI members with role/task assignments
- [SearchAugmentedLLM](https://github.com/EliasPereirah/SearchAugmentedLLM) – PHP search-augmented LLM tool that performs web search, extracts, chunks and ranks content to provide context for LLM responses (ideal for RAG applications)
- [skito/aipi-php](https://github.com/skito/aipi-php) – Universal API client for common AI models in PHP, offering a unified interface to interact with multiple LLM providers
- [takaaki-mizuno/php-llm-json-adapter](https://github.com/takaaki-mizuno/php-llm-json-adapter) — Adapter to normalize and return LLM responses as structured JSON using JSON Schema, with support for multiple providers (OpenAI, Gemini, Bedrock, Ollama)
- [thojou/php-llm-documents](https://github.com/thojou/php-llm-documents) — PHP library for LLM-based document processing (splitting, embeddings, vector store, search) inspired by LangChain/DocTran

### Tokenizers & Prompt Utilities

- [Gioni06/GPT3Tokenizer](https://github.com/Gioni06/GPT3Tokenizer) – PHP tokenizer compatible with GPT-3 style models
- [yethee/tiktoken-php](https://github.com/yethee/tiktoken-php) – PHP implementation of OpenAI’s *tiktoken* tokenizer for token counting and optimization

---

## Embeddings & Vector Search

*Libraries for generating embeddings and performing vector similarity search from PHP applications.*

- 🌟 [algolia/algoliasearch-client-php](https://github.com/algolia/algoliasearch-client-php) – ![GitHub stars](https://img.shields.io/github/stars/algolia/algoliasearch-client-php?style=social) Algolia search client
- [llm-agents-php/vector-storage](https://github.com/llm-agents-php/vector-storage) – LLM Agents Vector Storage
- 🌟 [LLPhant/LLPhant](https://github.com/LLPhant/LLPhant) – ![GitHub stars](https://img.shields.io/github/stars/LLPhant/LLPhant?style=social) Comprehensive PHP generative AI framework supporting LLMs, embeddings, vector search and more
- 🌟 [meilisearch/meilisearch-php](https://github.com/meilisearch/meilisearch-php) – ![GitHub stars](https://img.shields.io/github/stars/meilisearch/meilisearch-php?style=social) Client for Meilisearch search engine
- 🌟 [pgvector/pgvector](https://github.com/pgvector/pgvector) – ![GitHub stars](https://img.shields.io/github/stars/pgvector/pgvector?style=social) Vector similarity search extension for PostgreSQL
- 🌟 [pgvector/pgvector-php](https://github.com/pgvector/pgvector-php) – ![GitHub stars](https://img.shields.io/github/stars/pgvector/pgvector-php?style=social) PHP client for pgvector on PostgreSQL
- [redis-applied-ai/redis-vector-php](https://github.com/redis-applied-ai/redis-vector-php) — PHP client for Redis Vector Library (RedisVL) to support vector similarity search and AI-oriented queries
- [voyanara/milvus-php-sdk](https://github.com/voyanara/milvus-php-sdk) – PHP SDK for Milvus vector database API v2

---

## Data Processing

*ETL, data pipelines, serialization, and transformation utilities for preparing data for ML and analytics in PHP.*

- 🌟 [cocur/slugify](https://github.com/cocur/slugify) — ![GitHub stars](https://img.shields.io/github/stars/cocur/slugify?style=social) Converts strings into URL-friendly slugs, includes integrations for many frameworks
- 🌟 [flow-php/flow](https://github.com/flow-php/flow) – ![GitHub stars](https://img.shields.io/github/stars/flow-php/flow?style=social) Data processing and ETL framework for PHP with typed pipelines
- [league/csv](https://github.com/thephpleague/csv) – CSV data processing
- 🌟 [php-ds/ext-ds](https://github.com/php-ds/ext-ds) — ![GitHub stars](https://img.shields.io/github/stars/php-ds/ext-ds?style=social) PHP Data Structures extension: efficient vectors, maps, sets, etc.
- [spatie/data-transfer-object](https://github.com/spatie/data-transfer-object) – Strongly typed DTOs
- [symfony/serializer](https://github.com/symfony/serializer) – Data normalization & serialization

---

## Interop & Model Serving

*Bridging PHP with native libraries, external services, and runtimes for deploying and serving ML and LLM models.*

- 🌟 [ankane/onnxruntime-php](https://github.com/ankane/onnxruntime-php) – ![GitHub stars](https://img.shields.io/github/stars/ankane/onnxruntime-php?style=social) Run ONNX models from PHP
- 🌟 [deepl-php](https://github.com/DeepLcom/deepl-php) — ![GitHub stars](https://img.shields.io/github/stars/DeepLcom/deepl-php?style=social) Official PHP client library for the DeepL API, enabling high-quality language translation via DeepL’s AI/ML service
- [distantmagic/resonance](https://github.com/distantmagic/resonance) – Asynchronous PHP framework with built-in support for serving and integrating machine learning and LLM models
- [FFI](https://www.php.net/manual/en/book.ffi.php) – Native C/C++ bindings for ML inference
- 🧪 [garyblankenship/mcp-php](https://github.com/garyblankenship/mcp-php) – PHP example of a Model Context Protocol (MCP) server for connecting LLMs with application logic
- 🌟 [googleapis/google-cloud-php](https://github.com/googleapis/google-cloud-php) — ![GitHub stars](https://img.shields.io/github/stars/googleapis/google-cloud-php?style=social) Official PHP client library for Google Cloud APIs (including ML/AI services like Vision, Translate, AutoML, Vertex AI, etc.)
- [grpc/grpc-php](https://github.com/grpc/grpc-php) – gRPC client for model services
- 🌟 [neuron-core/neuron-ai](https://github.com/neuron-core/neuron-ai) – ![GitHub stars](https://img.shields.io/github/stars/neuron-core/neuron-ai?style=social) PHP agentic AI framework for building and orchestrating LLMs, RAG etc
- [nlpcloud/nlpcloud-php](https://github.com/nlpcloud/nlpcloud-php) — PHP client for the NLP Cloud API (access NLP/ML services like NER, sentiment analysis, summarization, text generation, embeddings, translation, and more)

---

## Tools & Utilities

*Supporting tools, debugging helpers, logging, and HTTP/CLI utilities commonly used in ML and AI workflows.*

- 🧪 [apphp/pretty-print](https://github.com/apphp/pretty-print) – Pretty-print PHP arrays and numeric data for ML debugging
- [guanguans/ai-commit](https://github.com/guanguans/ai-commit) — AI-powered CLI to automatically generate conventional Git commit messages
- [joshembling/laragenie](https://github.com/joshembling/laragenie) — AI chatbot/assistant for Laravel that indexes and understands your codebase via the command line (OpenAI + Pinecone)
- [nunomaduro/collision](https://github.com/nunomaduro/collision) – CLI error handling (useful for ML tools)
- [psr/log](https://github.com/php-fig/log) – Logging standard
- 🌟 [swoole/phpy](https://github.com/swoole/phpy) – ![GitHub stars](https://img.shields.io/github/stars/swoole/phpy?style=social) Bridge for calling Python from PHP via a runtime bridge
- [symfony/console](https://github.com/symfony/console) – CLI applications
- [symfony/http-client](https://github.com/symfony/http-client) – Robust HTTP client for AI APIs

---

## Laravel & Framework Integrations

### LLM & AI clients

- [artisan-build/llm](https://github.com/artisan-build/llm) — Laravel integration for multiple LLM providers (OpenAI, Azure, OpenRouter, etc.), simplifying usage of large language models in Laravel apps
- [BorahLabs/LLM-Port-Laravel](https://github.com/BorahLabs/LLM-Port-Laravel) – Laravel package for interchangeable LLM providers, allowing drop-in replacements of large language models
- [Capevace/llm-magic](https://github.com/Capevace/llm-magic) – Laravel-centric LLM toolkit with support for AI features like chat and structured data extraction
- [coding-wisely/taskallama](https://github.com/coding-wisely/taskallama) – Laravel package for seamless integration with the Ollama LLM API for AI-powered content generation, task assistance, conversation and embeddings
- [grok-php/laravel](https://github.com/grok-php/laravel) – Laravel package for integrating Grok AI models
- [moe-mizrak/laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter) – Laravel package to integrate OpenRouter LLM API
- 🌟 [neuron-core/neuron-laravel](https://github.com/neuron-core/neuron-laravel) – ![GitHub stars](https://img.shields.io/github/stars/neuron-core/neuron-laravel?style=social) Laravel integration for Neuron Core to build and orchestrate AI/LLM workflows
- 🌟 [openai-php/laravel](https://github.com/openai-php/laravel) – ![GitHub stars](https://img.shields.io/github/stars/openai-php/laravel?style=social) Laravel OpenAI integration 
- [PapaRascal2020/sidekick](https://github.com/PapaRascal2020/sidekick) – Laravel package offering a unified syntax for working with multiple AI provider APIs (OpenAI, Claude, Cohere, Mistral)
- [shawnveltman/laravel-openai](https://github.com/shawnveltman/laravel-openai) – Laravel wrapper for OpenAI
- [rahasistiyakofficial/laravel-ai-integration](https://github.com/rahasistiyakofficial/laravel-ai-integration) — This is a comprehensive, enterprise-ready package that provides seamless integration with multiple AI providers through a unified, elegant API

### Data & DTO tools

- [jeremysalmon/LaravelLLMContext](https://github.com/jeremysalmon/LaravelLLMContext) – Laravel package for managing and applying contextual data in LLM interactions
- [prism-php/prism](https://github.com/prism-php/prism) – Laravel interface for working with LLMs and AI providers
- [spatie/laravel-data](https://github.com/spatie/laravel-data) – Typed DTOs for API & AI responses

### Search & vector search

- 🌟 [laravel/scout](https://github.com/laravel/scout) – ![GitHub stars](https://img.shields.io/github/stars/laravel/scout?style=social) Search abstraction (useful for vector search)
- [teamtnt/laravel-scout-tntsearch-driver](https://github.com/teamtnt/laravel-scout-tntsearch-driver) – Local full-text search

---

## Symfony & Framework Integrations
- [openai-php/symfony](https://github.com/openai-php/symfony) – OpenAI PHP for Symfony integration
- 🌟 [symfony/ai](https://github.com/symfony/ai) – ![GitHub stars](https://img.shields.io/github/stars/symfony/ai?style=social) Symfony AI: built-in AI components and bundles for Symfony apps
    
---

## Resources

### General

- [Awesome PHP](https://github.com/ziadoz/awesome-php)

### Courses & Tutorials
- [Fun With OpenAI and Laravel](https://laracasts.com/series/fun-with-openai-and-laravel) – Laracasts series showing how to integrate OpenAI into Laravel apps

### ML / AI Platforms

- [ONNX Runtime](https://onnxruntime.ai) - Cross-platform, high performance ML inferencing and training accelerator
- [tensorflow/tfjs](https://github.com/tensorflow/tfjs) — JavaScript machine learning platform for training and running models in the browser or Node.js (TensorFlow.js)

### Learning Resources

- [Artificial Intelligence with PHP (GitBook)](https://apphp.gitbook.io/artificial-intelligence-with-php/) – Guide and reference for doing AI/ML with PHP
- 🌟 [AI для PHP-разработчиков: интуитивно и на практике (GitBook)](https://apphp.gitbook.io/ai-for-php-developers/) – Guide on AI with PHP in Russian / English
- [Build Your Own LLM in PHP (GitBook)](https://apphp.gitbook.io/build-your-own-llm-in-php/) – Guide to building an LLM from scratch in PHP
- [PHP FANN installation](https://www.php.net/manual/en/fann.installation.php) – Official PHP manual page for installing the FANN (Fast Artificial Neural Network) extension
- [PHP and LLMs (eBook)](https://leanpub.com/php_and_llms) – Practical book on integrating and using large language models with PHP
- [PHP-ML Tutorials](https://php-ml.readthedocs.io/en/latest/) - Documentation for PHP-ML for machine learning
- [Rubix ML Docs](https://rubixml.github.io/ML/latest/) - Comprehensive documentation for Rubix ML
 
---

## License

This list is licensed under the MIT License – see LICENSE for details.

## Contributing

Contributions are welcome!  
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details, including criteria for adding new projects (maintenance, documentation, tests, etc).

