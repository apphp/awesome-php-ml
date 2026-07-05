# Awesome PHP Machine Learning & AI

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub stars](https://img.shields.io/github/stars/apphp/awesome-php-ml?style=social)](https://github.com/apphp/awesome-php-ml)
[![Resources](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/apphp/awesome-php-ml/main/badge/resources.json)](https://github.com/apphp/awesome-php-ml#readme)
[![Last commit](https://img.shields.io/github/last-commit/apphp/awesome-php-ml)](https://github.com/apphp/awesome-php-ml/commits)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/apphp/awesome-php-ml/blob/main/LICENSE)
[![Link Check](https://github.com/apphp/awesome-php-ml/actions/workflows/link-check.yml/badge.svg)](https://github.com/apphp/awesome-php-ml/actions/workflows/link-check.yml)

The most comprehensive curated list of **Machine Learning, Artificial Intelligence, NLP, LLM and Data Science libraries for PHP**.

Inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and the broader **Awesome** ecosystem.

> **Goal:** make it easy to build intelligent systems with PHP — from classic ML to modern LLM-based workflows.

Want to add a project? See the [Contributing](#contributing) section below for inclusion criteria and submission guidance.

## Contents

- [Contents](#contents)
- [Requirements](#requirements)
- [What is this?](#what-is-this)
- [How to use this list](#how-to-use-this-list)
- [Quick Start](#quick-start)
- [Example "recipes"](#example-recipes)
- [Recommended core stack](#recommended-core-stack)
- [Legend](#legend)
- [Machine Learning](#machine-learning)
- [Deep Learning & Neural Networks](#deep-learning--neural-networks)
- [Natural Language Processing](#natural-language-processing)
- [Computer Vision, Image & Video Processing](#computer-vision-image--video-processing)
- [Math, Statistics & Linear Algebra](#math-statistics--linear-algebra)
- [Core ML Infrastructure](#core-ml-infrastructure)
- [LLMs & AI APIs](#llms--ai-apis)
- [Embeddings & Vector Search](#embeddings--vector-search)
- [Data Processing](#data-processing)
- [Interop & Model Serving](#interop--model-serving)
- [Tools & Utilities](#tools--utilities)
- [Laravel & Framework Integrations](#laravel--framework-integrations)
- [Symfony & Framework Integrations](#symfony--framework-integrations)
- [WordPress Integrations](#wordpress-integrations)
- [Resources](#resources)
- [Support this project](#support-this-project)

---

## Requirements

**PHP Version Requirements:**
- **Minimum**: PHP 7.4+ (most libraries already support PHP 8.1+/8.2+)
- **Recommended**: PHP 8.1+ for best performance and features
- **Latest features**: PHP 8.2+ for some cutting-edge libraries

**Common dependencies:**
- **Extensions**: `mbstring`, `curl`, `json`, `gd` (for image processing)
- **Optional**: `redis`, `pdo_pgsql` (for vector search), `ffi` (for native bindings)

**Memory considerations:**
- **Basic ML**: 256MB+ RAM
- **Neural networks**: 512MB+ RAM  
- **Large datasets**: 1GB+ RAM recommended

---

## What is this?

- Curated list of **PHP libraries and tools** for Machine Learning, AI, NLP, LLMs and Data Science.
- Focused on **code-first resources**: packages, SDKs, frameworks, and building blocks.
- Aimed at **PHP developers** who want to add intelligent features to existing apps or build new AI-powered systems.

## How to use this list

- **Classic ML / traditional models** – start with [php-ai/php-ml](https://gitlab.com/php-ai/php-ml) and [RubixML/RubixML](https://github.com/RubixML/RubixML).
- **LLM-powered apps & agents** – see [LLMs & AI APIs](#llms--ai-apis), [Embeddings & Vector Search](#embeddings--vector-search), and framework integrations (Laravel/Symfony).
- **RAG (Retrieval-Augmented Generation)** – combine [php-rag](https://github.com/mzarnecki/php-rag) with vector databases like [pgvector](https://github.com/pgvector/pgvector) or [Meilisearch](https://github.com/meilisearch/meilisearch-php).
- **Numerical computing & math** – explore [Core ML Infrastructure](#core-ml-infrastructure) for tensors and matrices, and [Math, Statistics & Linear Algebra](#math-statistics--linear-algebra) for statistics and related math.
- **Production integration** – use [Interop & Model Serving](#interop--model-serving) and framework integrations to wire models into real apps.

### Quick Start

**For beginners new to PHP ML/AI:**

```bash
# Install a core ML library
composer require rubix/ml
composer require php-ai/phpml

# Install LLM client
composer require openai-php/client

# Install vector search for RAG
composer require llphant/llphant
```

**Basic examples:**
- **Classification**: Use `RubixML/RubixML` with `KNearestNeighbors` for simple classification tasks
- **LLM integration**: Use `openai-php/client` to call GPT models from PHP
- **Text analysis**: Use `php-ai/php-ml` for sentiment analysis and tokenization
- **Vector search**: Use `LLPhant/LLPhant` with `pgvector` for semantic search

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

Not all projects are tagged yet – we're gradually adding markers as the ecosystem evolves. Treat them as rough guidance, not strict rules.

- `🌟` – widely used / production-ready projects
- `🧪` – experimental or research-oriented projects
- `⚠️` – projects with limited maintenance, older APIs, or niche usage; review before using in new projects

---

## Machine Learning

*Core PHP libraries for supervised/unsupervised learning, classification, regression, and clustering.*

- 🌟 [CodeWithKyrian/transformers-php](https://github.com/CodeWithKyrian/transformers-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/CodeWithKyrian/transformers-php?style=social) A PHP toolkit for running Hugging Face–style Transformer models with ONNX Runtime (text generation, summarization, classification, etc.)
- ⚠️ [danielefavi/brainy](https://github.com/danielefavi/brainy "Link to resource") – Simple PHP class for neural networks and machine learning
- [dr-que/polynomial-regression](https://github.com/jbboehr/PolynomialRegression.php "Link to resource") – Polynomial regression for PHP
- ⚠️ [pecl/svm](https://pecl.php.net/package/svm/0.2.3 "Link to resource") – PHP extension providing bindings to the LIBSVM library for Support Vector Machine classification and regression
- 🌟 [php-ai/php-ml](https://gitlab.com/php-ai/php-ml "Link to resource") – Core machine learning algorithms for PHP
- [php-ai/php-ml-examples](https://github.com/php-ai/php-ml-examples "Link to resource") – Practical examples for PHP-ML
- [sphamster/bayes](https://github.com/sphamster/bayes "Link to resource") – Naive Bayes classifier implementation in PHP for probabilistic classification tasks

---

## Deep Learning & Neural Networks

*PHP libraries for neural networks, deep learning architectures, and advanced learners built on tensors.*

- 🧪 [rindow/rindow-neuralnetworks](https://github.com/rindow/rindow-neuralnetworks "Link to resource") – Deep learning framework for PHP providing neural network layers, training utilities, and GPU/accelerated backends via the Rindow numerical computing ecosystem
- 🌟 [RubixML/RubixML](https://github.com/RubixML/RubixML "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/RubixML/RubixML?style=social) High-level ML framework with pipelines and datasets

---

## Natural Language Processing

*Text processing, tokenization, language detection, sentiment analysis and other NLP tasks in PHP.*

- ⚠️ [angeloskath/php-nlp-tools](https://github.com/angeloskath/php-nlp-tools "Link to resource") – Natural Language Processing tools
- 🌟 [ankane/mitie-php](https://github.com/ankane/mitie-php "Link to resource") – PHP bindings for the MITIE NLP library providing named entity recognition (NER), text classification, and feature extraction using pre-trained statistical models
- [davmixcool/php-sentiment-analyzer](https://github.com/davmixcool/php-sentiment-analyzer "Link to resource") – Lightweight PHP library for sentiment analysis using lexical rules
- [friteuseb/nlp_tools](https://github.com/friteuseb/nlp_tools "Link to resource") – Extension for NLP methods and text analysis
- ⚠️ [googlei18n/myanmar-tools](https://github.com/googlei18n/myanmar-tools "Link to resource") – Myanmar text encoding detection and Zawgyi ↔ Unicode conversion using a trained model (includes PHP support)
- ⚠️ [patrickschur/language-detection](https://github.com/patrickschur/language-detection "Link to resource") – Language detection library
- 🧪 [RubixML/Sentiment](https://github.com/RubixML/Sentiment "Link to resource") – Example project demonstrating sentiment analysis with a neural network (IMDB reviews) using Rubix ML in PHP
- 🧪 [SerafimArts/TF-IDF](https://github.com/SerafimArts/TF-IDF "Link to resource") – Simple TF-IDF implementation for keyword extraction and text relevance scoring in PHP
- [voku/stop-words](https://github.com/voku/stop-words "Link to resource") – Stop word lists for many languages
- [yooper/php-text-analysis](https://github.com/yooper/php-text-analysis "Link to resource") – Sentiment analysis and NLP tools

---

## Computer Vision, Image & Video Processing

*Image manipulation, preprocessing, and computer vision workloads from PHP.*

- 🧪 [aschmelyun/subvert](https://github.com/aschmelyun/subvert "Link to resource") - Generate subtitles, summaries, and chapters from videos in seconds
- 🌟 [Intervention/image](https://github.com/Intervention/image "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/Intervention/image?style=social) Image manipulation library for CV preprocessing
- [jcupitt/vips](https://github.com/jcupitt/libvips "Link to resource") – Fast image processing library with PHP bindings
- 🧪 [mailmug/php-dlib](https://github.com/mailmug/php-dlib "Link to resource") – PHP extension for Dlib, supporting face detection, facial landmarks, face recognition descriptors, CNN detection, and clustering
- 🧪 [php-opencv/php-opencv](https://github.com/php-opencv/php-opencv "Link to resource") – OpenCV bindings for PHP

---

## Math, Statistics & Linear Algebra

*Numerical computing, matrix operations, statistics, and related math foundations for ML and data science in PHP.*

- 🌟 [brick/math](https://github.com/brick/math "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/brick/math?style=social) Arbitrary-precision arithmetic for PHP (BigInteger, BigDecimal, BigRational)
- 🌟 [Hi-Folks/statistics](https://github.com/Hi-Folks/statistics "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/Hi-Folks/statistics?style=social) Probability distributions and statistical functions library for PHP
- 🌟 [markrogoyski/math-php](https://github.com/markrogoyski/math-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/markrogoyski/math-php?style=social) Math library for linear algebra, statistics, and calculus
- [mcordingley/LinearAlgebra](https://github.com/mcordingley/LinearAlgebra "Link to resource") – Stand-alone linear algebra library
- ⚠️ [NumPHP/NumPHP](https://github.com/NumPHP/NumPHP "Link to resource") – Math library for scientific computing

---

## Core ML Infrastructure

*Low-level building blocks for numerical computing, tensors, and model execution in PHP.*

### Numerical computing & tensors

- 🌟 [RubixML/Tensor](https://github.com/RubixML/Tensor "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/RubixML/Tensor?style=social) N-dimensional tensors for numerical computing
- 🌟 [RubixML/numpower](https://github.com/RubixML/numpower "Link to resource") – High-performance numerical computing library inspired by NumPy
- 🧪 [rindow/rindow-math-matrix](https://github.com/rindow/rindow-math-matrix "Link to resource") – Foundational package for scientific matrix operations
- 🧪 [phpmlkit/ndarray](https://github.com/phpmlkit/ndarray "Link to resource") – Multidimensional array (ndarray) implementation for PHP inspired by NumPy, useful for numerical computing and machine learning workloads
- 🌟 [krakjoe/ort](https://github.com/krakjoe/ort "Link to resource") – – ![GitHub stars](https://img.shields.io/github/stars/krakjoe/ort?style=social) PHP extension for high-performance tensor mathematics, with optional ONNX Runtime integration for model inference

### Model execution & runtimes

- 🌟 [ankane/onnxruntime-php](https://github.com/ankane/onnxruntime-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/ankane/onnxruntime-php?style=social) Run ONNX models from PHP
- [FFI](https://www.php.net/manual/en/book.ffi.php "Link to resource") – Native C/C++ bindings in PHP for high-performance ML inference
- [phpmlkit/onnxruntime](https://github.com/phpmlkit/onnxruntime "Link to resource") – High-performance ONNX Runtime bindings for PHP using FFI, enabling inference of models from PyTorch, TensorFlow, scikit-learn and other frameworks

### Interoperability

- 🌟 [swoole/phpy](https://github.com/swoole/phpy "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/swoole/phpy?style=social) Bridge for calling Python from PHP via a runtime bridge

### Ecosystems

- [phpmlkit (GitHub org)](https://github.com/phpmlkit "Link to resource") – Collection of high-performance ML infrastructure libraries for PHP, including NDArray (NumPy-like arrays) and ONNX Runtime bindings

---

## LLMs & AI APIs

*Clients, SDKs, and frameworks for calling hosted LLMs and other AI providers from PHP.*

- [aiaccess/ai-access](https://github.com/aiaccess/ai-access "Link to resource") – Unified PHP AI client providing a consistent interface for multiple providers (OpenAI, Anthropic, Gemini, DeepSeek, Grok) with support for chat, embeddings, batch processing, and provider switching
- [aimeos/prisma](https://github.com/aimeos/prisma "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/aimeos/prisma?style=social) Lightweight PHP package providing a unified interface for text, image, audio, and video AI providers
- 🧪 [adrienbrault/instructrice](https://github.com/adrienbrault/instructrice "Link to resource") – Typed LLM outputs in PHP with flexible schema support (OpenAI, Claude, Gemini, etc.) and type-safe handling of structured responses
- [ArdaGnsrn/ollama-php](https://github.com/ArdaGnsrn/ollama-php "Link to resource") – A PHP client library for the Ollama LLM server, enabling completions, chat, model management, and embeddings via Ollama's API
- [Clarifai/clarifai-php-grpc](https://github.com/Clarifai/clarifai-php-grpc "Link to resource") – Official Clarifai gRPC PHP client for accessing Clarifai's AI APIs (vision and text recognition)
- [cognesy/instructor-php](https://github.com/cognesy/instructor-php "Link to resource") – Structured-output helper for LLM responses
- [deepseek-php/deepseek-php-client](https://github.com/deepseek-php/deepseek-php-client "Link to resource") – PHP client library for integrating with the DeepSeek AI API, providing a fluent API for model queries, streaming results, and support for multiple HTTP clients and models
- 🌟 [dtyq/magic](https://github.com/dtyq/magic "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/dtyq/magic?style=social) Open-source enterprise AI agent platform with generalist agents, workflow orchestration, IM integration, collaborative office features, and support for multiple LLMs
- [elastic/elasticsearch-chatgpt-php](https://github.com/elastic/elasticsearch-chatgpt-php "Link to resource") – Experimental PHP library that uses ChatGPT to translate natural language into Elasticsearch DSL queries and perform semantic search over your indices
- [FunkyOz/mulagent](https://github.com/FunkyOz/mulagent "Link to resource") – Multi-agent orchestration framework for LLM applications
- 🌟 [google-gemini-php/client](https://github.com/google-gemini-php/client "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/google-gemini-php/client?style=social) Gemini PHP is a community-maintained PHP API client that allows you to interact with the Gemini AI API
- ⚠️ [HosonoDE/EasyAI-PHP](https://github.com/HosonoDE/EasyAI-PHP "Link to resource") – High-level AI integration library for PHP that simplifies using LLMs
- 🧪 [carmelosantana/php-agents](https://github.com/carmelosantana/php-agents) – PHP framework for building AI agents with tool use, provider abstraction and multi-model support
- 🌟 [kambo-1st/langchain-php](https://github.com/kambo-1st/langchain-php "Link to resource") ![GitHub stars](https://img.shields.io/github/stars/kambo-1st/langchain-php?style=social) A PHP port of the LangChain framework for building composable LLM-powered applications
- 🌟 [llm-agents-php/agents](https://github.com/llm-agents-php/agents "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/llm-agents-php/agents?style=social) LM Agents is a PHP library for building and managing Language Model (LLM) based agents
- [llm-agents-php/prompt-generator](https://github.com/llm-agents-php/prompt-generator "Link to resource") – Prompt generator for LLM agents with interceptors
- [ModelFlow-AI (GitHub org)](https://github.com/modelflow-ai "Link to resource") – Collection of PHP packages for unified access to AI models, embeddings, and chat (OpenAI, Mistral, Ollama)
- [mozex/anthropic-php](https://github.com/mozex/anthropic-php "Link to resource") – Community-maintained PHP API client for the Anthropic (Claude) AI API, supporting messages, streaming, tool use, and batch processing
- [mzarnecki/php-rag](https://github.com/mzarnecki/php-rag "Link to resource") – PHP RAG toolkit for connecting vector search and LLMs in retrieval-augmented workflows
- 🌟 [openai-php/client](https://github.com/openai-php/client "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/openai-php/client?style=social) Official OpenAI PHP client
- 🌟 [orhanerday/open-ai](https://github.com/orhanerday/open-ai "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/orhanerday/open-ai?style=social) Popular OpenAI PHP SDK
- [prism-php/bedrock](https://github.com/prism-php/bedrock "Link to resource") – AWS Bedrock provider for the Prism PHP framework, adding Bedrock LLM and embeddings support to Laravel Prism integrations
- [sarfraznawaz2005/ai-team](https://github.com/sarfraznawaz2005/ai-team "Link to resource") – Package to build and run collaborative teams of AI members with role/task assignments
- [SearchAugmentedLLM](https://github.com/EliasPereirah/SearchAugmentedLLM "Link to resource") – PHP search-augmented LLM tool that performs web search, extracts, chunks and ranks content to provide context for LLM responses (ideal for RAG applications)
- [skito/aipi-php](https://github.com/skito/aipi-php "Link to resource") – Universal API client for common AI models in PHP, offering a unified interface to interact with multiple LLM providers
- [softcreatr/php-mistral-ai-sdk](https://github.com/SoftCreatR/php-mistral-ai-sdk "Link to resource") – PHP SDK for the Mistral AI API, providing an easy wrapper to call Mistral's LLM and AI endpoints (chat, embeddings, fine-tuning etc.)
- [takaaki-mizuno/php-llm-json-adapter](https://github.com/takaaki-mizuno/php-llm-json-adapter "Link to resource") – Adapter to normalize and return LLM responses as structured JSON using JSON Schema, with support for multiple providers (OpenAI, Gemini, Bedrock, Ollama)
- [thojou/php-llm-documents](https://github.com/thojou/php-llm-documents "Link to resource") – PHP library for LLM-based document processing (splitting, embeddings, vector store, search) inspired by LangChain/DocTran
- [utopia-php/agents](https://github.com/utopia-php/agents "Link to resource") – Simple, lightweight PHP library for AI agent orchestration with multi-provider support (OpenAI, Anthropic, Deepseek, Perplexity, XAI)

### Agents & Tooling / MCP

- [logiscape/mcp-sdk-php](https://github.com/logiscape/mcp-sdk-php "Link to resource") – PHP SDK for building Model Context Protocol (MCP) clients and servers to connect LLMs with external tools and services
- 🧪 [manuelkiessling/php-ai-tool-bridge](https://github.com/manuelkiessling/php-ai-tool-bridge "Link to resource") – PHP library for defining AI “tool functions” that let LLMs interact with application code and external services using structured JSON schemas
- 🧪 [neuron-core/youtube-ai-agent](https://github.com/neuron-core/youtube-ai-agent "Link to resource") – Example PHP AI agent built with Neuron for summarizing YouTube videos and generating content from them
- [prism-php/relay](https://github.com/prism-php/relay "Link to resource") – MCP client for Prism that lets PHP/Laravel AI agents connect to external Model Context Protocol servers and use their tools
- 🧪 [symfony/mcp-sdk](https://github.com/symfony/mcp-sdk "Link to resource") – Symfony's experimental PHP SDK for building Model Context Protocol (MCP) clients and servers

### Speech & Text-to-Speech

- [b7s/fluentvox](https://github.com/b7s/fluentvox "Link to resource") – Fluent PHP API for state-of-the-art text-to-speech and voice cloning (Resemble AI's Chatterbox), with CLI, GPU acceleration, and multilingual support
- [b7s/whisper-php](https://github.com/b7s/whisper-php "Link to resource") – PHP wrapper/client for Whisper speech-to-text (ASR), enabling audio transcription via Whisper models

### Tokenizers & Prompt Utilities

- [CodeWithKyrian/tokenizers-php](https://github.com/CodeWithKyrian/tokenizers-php "Link to resource") – PHP bindings for Hugging Face Tokenizers, enabling fast tokenization for transformer and LLM models
- [Gioni06/GPT3Tokenizer](https://github.com/Gioni06/GPT3Tokenizer "Link to resource") – PHP tokenizer compatible with GPT-3 style models
- [HelgeSverre/toon-php](https://github.com/HelgeSverre/toon-php "Link to resource") – PHP implementation of TOON, a compact data format for reducing token usage when sending structured data to LLMs
- [RahulDey12/tiktoken-php](https://github.com/RahulDey12/tiktoken-php "Link to resource") – PHP implementation of OpenAI's BPE tokenizer `tiktoken` for encoding, decoding, and counting tokens in GPT prompts
- [yethee/tiktoken-php](https://github.com/yethee/tiktoken-php "Link to resource") – PHP implementation of OpenAI's *tiktoken* tokenizer for token counting and optimization

---

## Embeddings & Vector Search

*Libraries for generating embeddings and performing vector similarity search from PHP applications.*

- 🌟 [algolia/algoliasearch-client-php](https://github.com/algolia/algoliasearch-client-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/algolia/algoliasearch-client-php?style=social) Algolia search client
- [CodeWithKyrian/chromadb-php](https://github.com/CodeWithKyrian/chromadb-php "Link to resource") – PHP client for ChromaDB, enabling vector similarity search and embedding storage for AI and RAG applications
- [hkulekci/qdrant-php](https://github.com/hkulekci/qdrant-php "Link to resource") – PHP client for the Qdrant vector database, enabling vector similarity search and embedding storage for AI and RAG applications
- [llm-agents-php/vector-storage](https://github.com/llm-agents-php/vector-storage "Link to resource") – LLM Agents Vector Storage
- 🌟 [LLPhant/LLPhant](https://github.com/LLPhant/LLPhant "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/LLPhant/LLPhant?style=social) Comprehensive PHP generative AI framework supporting LLMs, embeddings, vector search and more
- 🌟 [meilisearch/meilisearch-php](https://github.com/meilisearch/meilisearch-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/meilisearch/meilisearch-php?style=social) Client for Meilisearch search engine
- 🌟 [pgvector/pgvector](https://github.com/pgvector/pgvector "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/pgvector/pgvector?style=social) Vector similarity search extension for PostgreSQL
- 🌟 [pgvector/pgvector-php](https://github.com/pgvector/pgvector-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/pgvector/pgvector-php?style=social) PHP client for pgvector on PostgreSQL
- [probots-io/pinecone-php](https://github.com/probots-io/pinecone-php) – PHP client for Pinecone vector database used in semantic search and RAG pipelines
- [redis-applied-ai/redis-vector-php](https://github.com/redis-applied-ai/redis-vector-php "Link to resource") – PHP client for Redis Vector Library (RedisVL) to support vector similarity search and AI-oriented queries
- [voyanara/milvus-php-sdk](https://github.com/voyanara/milvus-php-sdk "Link to resource") – PHP SDK for Milvus vector database API v2

---

## Data Processing

*ETL, data pipelines, serialization, and transformation utilities for preparing data for ML and analytics in PHP.*

- 🌟 [cocur/slugify](https://github.com/cocur/slugify "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/cocur/slugify?style=social) Converts strings into URL-friendly slugs, includes integrations for many frameworks
- 🌟 [flow-php/flow](https://github.com/flow-php/flow "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/flow-php/flow?style=social) Data processing and ETL framework for PHP with typed pipelines
- [league/csv](https://github.com/thephpleague/csv "Link to resource") – CSV data processing
- [paperdoc-dev/paperdoc-lib](https://github.com/paperdoc-dev/paperdoc-lib "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/paperdoc-dev/paperdoc-lib?style=social) Zero-dependency PHP library for generating, parsing, and converting documents such as PDF, HTML, CSV, DOCX, XLSX, PPTX, and Markdown
- 🌟 [php-ds/ext-ds](https://github.com/php-ds/ext-ds "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/php-ds/ext-ds?style=social) PHP Data Structures extension: efficient vectors, maps, sets, etc.
- [spatie/data-transfer-object](https://github.com/spatie/data-transfer-object "Link to resource") – Strongly typed DTOs
- [symfony/serializer](https://github.com/symfony/serializer "Link to resource") – Data normalization & serialization

---

## Interop & Model Serving

*Bridging PHP with native libraries, external services, and runtimes for deploying and serving ML and LLM models.*

- 🌟 [deepl-php](https://github.com/DeepLcom/deepl-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/DeepLcom/deepl-php?style=social) Official PHP client library for the DeepL API, enabling high-quality language translation via DeepL's AI/ML service
- [distantmagic/resonance](https://github.com/distantmagic/resonance "Link to resource") – Asynchronous PHP framework (Swoole-based) for building AI-powered, IO-intensive applications, with built-in web server, LLM integration (llama.cpp), WebSockets, and ML model serving capabilities
- [FFI](https://www.php.net/manual/en/book.ffi.php "Link to resource") – Native C/C++ bindings for ML inference
- 🧪 [garyblankenship/mcp-php](https://github.com/garyblankenship/mcp-php "Link to resource") – PHP example of a Model Context Protocol (MCP) server for connecting LLMs with application logic
- 🌟 [googleapis/google-cloud-php](https://github.com/googleapis/google-cloud-php "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/googleapis/google-cloud-php?style=social) Official PHP client library for Google Cloud APIs (including ML/AI services like Vision, Translate, AutoML, Vertex AI, etc.)
- [grpc/grpc-php](https://github.com/grpc/grpc-php "Link to resource") – gRPC client for model services
- 🧪 [HossamBalaha/Deep-Learning-Classification-System-using-PHP-and-Keras](https://github.com/HossamBalaha/Deep-Learning-Classification-System-using-PHP-and-Keras "Link to resource") – Example system showing how to integrate a Keras deep learning classifier with a PHP backend
- 🌟 [neuron-core/neuron-ai](https://github.com/neuron-core/neuron-ai "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/neuron-core/neuron-ai?style=social) PHP agentic AI framework for building and orchestrating LLMs, RAG etc
- [nlpcloud/nlpcloud-php](https://github.com/nlpcloud/nlpcloud-php "Link to resource") – PHP client for the NLP Cloud API (access NLP/ML services like NER, sentiment analysis, summarization, text generation, embeddings, translation, and more)
---

## Tools & Utilities

*Supporting tools, debugging helpers, logging, and HTTP/CLI utilities commonly used in ML and AI workflows.*

- 🧪 [apphp/pretty-print](https://github.com/apphp/pretty-print "Link to resource") – Pretty-print PHP arrays and numeric data for ML debugging
- 🧪 [context-hub/generator](https://github.com/context-hub/generator "Link to resource") – Context-as-Code (CTX) tool that extracts and organizes codebase context into structured documents and MCP servers for LLM-assisted development
- [guanguans/ai-commit](https://github.com/guanguans/ai-commit "Link to resource") – AI-powered CLI to automatically generate conventional Git commit messages
- 🧪 [hiblaphp/http-client](https://github.com/hiblaphp/http-client "Link to resource") – Lightweight PSR-7/PSR-18 compatible HTTP client for interacting with AI APIs and external services from PHP
- [joshembling/laragenie](https://github.com/joshembling/laragenie "Link to resource") – AI chatbot/assistant for Laravel that indexes and understands your codebase via the command line (OpenAI + Pinecone)
- 🧪 [mariorazo97/single-file-php-ai](https://github.com/mariorazo97/single-file-php-ai "Link to resource") – Drop-in single-file PHP AI chat interface for Ollama and OpenAI, with no Node.js, Docker, database, or build step
- [nunomaduro/collision](https://github.com/nunomaduro/collision "Link to resource") – CLI error handling (useful for ML tools)
- [psr/log](https://github.com/php-fig/log "Link to resource") – Logging standard
- [symfony/console](https://github.com/symfony/console "Link to resource") – CLI applications
- [symfony/http-client](https://github.com/symfony/http-client "Link to resource") – Robust HTTP client for AI APIs

---

## Laravel & Framework Integrations

### LLM & AI clients

- [artisan-build/llm](https://github.com/artisan-build/llm "Link to resource") – Laravel integration for multiple LLM providers (OpenAI, Azure, OpenRouter, etc.), simplifying usage of large language models in Laravel apps
- 🧪 [builtbyberry/laravel-swarm](https://github.com/builtbyberry/laravel-swarm "Link to resource") – Multi-agent swarm orchestration for Laravel, built on Laravel AI, with sequential, parallel, hierarchical, queued, streamed, and durable workflows
- [BorahLabs/LLM-Port-Laravel](https://github.com/BorahLabs/LLM-Port-Laravel "Link to resource") – Laravel package for interchangeable LLM providers, allowing drop-in replacements of large language models
- [Capevace/llm-magic](https://github.com/Capevace/llm-magic "Link to resource") – Laravel-centric LLM toolkit with support for AI features like chat and structured data extraction
- [coding-wisely/taskallama](https://github.com/coding-wisely/taskallama "Link to resource") – Laravel package for seamless integration with the Ollama LLM API for AI-powered content generation, task assistance, conversation and embeddings
- [grok-php/laravel](https://github.com/grok-php/laravel "Link to resource") – Laravel package for integrating Grok AI models
- 🌟 [laravel/ai](https://github.com/laravel/ai "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/laravel/ai?style=social) The Laravel AI SDK: a unified, expressive Laravel API for interacting with AI providers (LLMs, images, embeddings, agents, tools)
- 🌟 [laravel/boost](https://github.com/laravel/boost "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/laravel/boost?style=social) Official Laravel Boost: a development server and AI context provider that accelerates AI-assisted code generation by giving AI tools detailed insight into your Laravel app (MCP server, schema inspection, docs + guidelines)
- [maestroerror/LarAgent](https://github.com/maestroerror/LarAgent "Link to resource") – AI agent development framework for Laravel: define agents, tools, workflows, and manage LLM interactions with an Eloquent-style API
- [moe-mizrak/laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter "Link to resource") – Laravel package to integrate OpenRouter LLM API
- [mozex/anthropic-laravel](https://github.com/mozex/anthropic-laravel "Link to resource") – Laravel integration for the Anthropic (Claude) AI API with Facades, config publishing, and testing fakes
- 🌟 [neuron-core/neuron-laravel](https://github.com/neuron-core/neuron-laravel "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/neuron-core/neuron-laravel?style=social) Laravel integration for Neuron Core to build and orchestrate AI/LLM workflows
- [atlas-php/atlas](https://github.com/atlas-php/atlas "Link to resource") – Laravel AI application framework for structuring agents, tools, prompts, and pipelines on top of Prism PHP
- 🌟 [openai-php/laravel](https://github.com/openai-php/laravel "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/openai-php/laravel?style=social) Laravel OpenAI integration 
- [opgginc/laravel-mcp-server](https://github.com/opgginc/laravel-mcp-server "Link to resource") – Laravel package for building secure Model Context Protocol (MCP) servers using Streamable HTTP/SSE, enabling real-time communication between LLM agents and application tools
- [PapaRascal2020/sidekick](https://github.com/PapaRascal2020/sidekick "Link to resource") – Laravel package offering a unified syntax for working with multiple AI provider APIs (OpenAI, Claude, Cohere, Mistral)
- 🌟 [php-mcp/laravel](https://github.com/php-mcp/laravel "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/php-mcp/laravel?style=social) – Laravel package for building Model Context Protocol (MCP) servers and exposing application tools to LLMs
- 🌟 [promptlyagentai/promptlyagent](https://github.com/promptlyagentai/promptlyagent "Link to resource") – AI Agent development framework / workbench / harness powered by Laravel
- [shawnveltman/laravel-openai](https://github.com/shawnveltman/laravel-openai "Link to resource") – Laravel wrapper for OpenAI
- [vizra-ai/vizra-adk](https://github.com/vizra-ai/vizra-adk "Link to resource") – Laravel AI Agent Development Kit for building autonomous agents with tools, persistent memory, workflows, streaming, evaluations, tracing, and Prism-powered multi-model support
- [rahasistiyakofficial/laravel-ai-integration](https://github.com/rahasistiyakofficial/laravel-ai-integration "Link to resource") – This is a comprehensive, enterprise-ready package that provides seamless integration with multiple AI providers through a unified, elegant API

### Data & DTO tools

- [jeremysalmon/LaravelLLMContext](https://github.com/jeremysalmon/LaravelLLMContext "Link to resource") – Laravel package for managing and applying contextual data in LLM interactions
- 🌟 [prism-php/prism](https://github.com/prism-php/prism "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/prism-php/prism?style=social) Unified Laravel-native interface for working with LLMs (OpenAI, Anthropic, Gemini, Ollama, etc.), supporting text generation, structured outputs, tools/function calling, and multi-step AI workflows
- [spatie/laravel-data](https://github.com/spatie/laravel-data "Link to resource") – Typed DTOs for API & AI responses

### Localization & Translation

- [Capevace/ai-translations-for-laravel](https://github.com/Capevace/ai-translations-for-laravel "Link to resource") – Laravel package for automatically translating language files with LLMs, detecting missing translations, updating existing locales, validating translation files, and refining translations interactively
- [jayeshmepani/laravel-gemini-translator](https://github.com/jayeshmepani/laravel-gemini-translator "Link to resource") – Laravel Gemini AI Translation Extractor scans your Laravel project for translation keys, uses Google Gemini AI for translations, and generates language files automatically — streamlining and accelerating your localization workflow

### Monitoring / Cost Control

- 🧪 [subhashladumor1/laravel-ai-guard](https://github.com/subhashladumor1/laravel-ai-guard "Link to resource") – Laravel package for tracking LLM token usage, estimating AI costs, enforcing per-user or per-tenant budgets, and preventing unexpected AI billing spikes

### MCP / Tooling

- [RedberryProducts/mcp-client-laravel](https://github.com/RedberryProducts/mcp-client-laravel "Link to resource") – Laravel-native MCP client for connecting to Model Context Protocol servers via HTTP or STDIO, retrieving tools and resources, and integrating external agent capabilities into Laravel apps

### Prompt Management

- 🧪 [prismaticoder/laravel-prompt-manager](https://github.com/prismaticoder/laravel-prompt-manager "Link to resource") – Laravel package for managing, versioning, and testing AI prompts for LLM-powered applications
- 🧪 [SabatinoMasala/laravel-llm-prompt](https://github.com/SabatinoMasala/laravel-llm-prompt "Link to resource") – Lightweight Laravel helper for defining, templating, and composing LLM prompts using PHP classes with variable interpolation and dynamic prompt building

### Search & vector search

- 🌟 [laravel/scout](https://github.com/laravel/scout "Link to resource") – ![GitHub stars](https://img.shields.io/github/stars/laravel/scout?style=social) – Search abstraction (useful for vector search)
- [teamtnt/laravel-scout-tntsearch-driver](https://github.com/teamtnt/laravel-scout-tntsearch-driver "Link to resource") – Local full-text search

---

## Symfony & Framework Integrations

- [openai-php/symfony](https://github.com/openai-php/symfony "Link to resource") – OpenAI PHP for Symfony integration
- 🌟 [symfony/ai](https://github.com/symfony/ai) – ![GitHub stars](https://img.shields.io/github/stars/symfony/ai?style=social "Link to resource") – Symfony AI: built-in AI components and bundles for Symfony apps
- [soleinjast/symfony-markdown-response-bundle](https://github.com/soleinjast/symfony-markdown-response-bundle "Link to resource") – Symfony bundle that automatically serves Markdown versions of HTML responses to clients
- 🧪 [symfony/ai-agent](https://github.com/symfony/ai-agent "Link to resource") – Symfony AI Agent component for building agentic applications that interact with users, execute tasks, and manage workflows
- 🧪 [symfony/ai-bundle](https://github.com/symfony/ai-bundle "Link to resource") – Symfony integration bundle that brings together Symfony AI components for agents, chat, platforms, stores, RAG, tools, and configuration
- 🧪 [symfony/ai-platform](https://github.com/symfony/ai-platform "Link to resource") – Experimental Symfony AI Platform component providing a unified abstraction for interacting with AI models, providers, messages, embeddings, speech, and provider-specific bridge packages
- [symfony/ai-store](https://github.com/symfony/ai-store "Link to resource") – Symfony AI component providing a vector store abstraction for semantic search and RAG workflows
- [symfony/mcp-bundle](https://github.com/symfony/mcp-bundle "Link to resource") – Symfony bundle for exposing MCP tools, prompts, and resources over HTTP or STDIO using the official MCP SDK

---

## WordPress Integrations

- [WordPress/php-ai-client](https://github.com/WordPress/php-ai-client "Link to resource") – Provider-agnostic PHP AI SDK offering a unified API for interacting with multiple LLM providers (OpenAI, Anthropic, Gemini, etc.), supporting text, image, speech, streaming, and multimodal operations

---

## Resources

### General

- [Awesome PHP](https://github.com/ziadoz/awesome-php "Link to resource")
- 🧪 [dykyi-roman/awesome-claude-code](https://github.com/dykyi-roman/awesome-claude-code "Link to resource") – Curated collection of commands, agents, skills, hooks, and tools for enhancing Claude Code AI workflows

### Courses & Tutorials

- [Fun With OpenAI and Laravel](https://laracasts.com/series/fun-with-openai-and-laravel "Link to resource") – Laracasts series showing how to integrate OpenAI into Laravel apps
- [Laravel Cloud Skills](https://skills.laravel.cloud) – Interactive learning platform for building and deploying Laravel applications, including modern AI and cloud workflows

### ML / AI Platforms

- [ONNX Runtime](https://onnxruntime.ai "Link to resource") – Cross-platform, high performance ML inferencing and training accelerator
- [tensorflow/tfjs](https://github.com/tensorflow/tfjs "Link to resource") – JavaScript machine learning platform for training and running models in the browser or Node.js (TensorFlow.js)

### Learning Resources

- [Artificial Intelligence with PHP (GitBook)](https://apphp.gitbook.io/artificial-intelligence-with-php/ "Link to resource") – Guide and reference for doing AI/ML with PHP
- 🌟 [AI for PHP Developers: Intuitive and Practical (GitBook)](https://apphp.gitbook.io/ai-for-php-developers/ "Link to resource") – Guide on AI with PHP in Russian / English
- [Build Your Own LLM in PHP (GitBook)](https://apphp.gitbook.io/build-your-own-llm-in-php/ "Link to resource") – Guide to building an LLM from scratch in PHP
- [PHP FANN installation](https://www.php.net/manual/en/fann.installation.php "Link to resource") – Official PHP manual page for installing the FANN (Fast Artificial Neural Network) extension
- [PHP and LLMs (eBook)](https://leanpub.com/php_and_llms "Link to resource") – Practical book on integrating and using large language models with PHP
- [PHP-ML Tutorials](https://php-ml.readthedocs.io/en/latest/ "Link to resource") – Documentation for PHP-ML for machine learning
- [Rubix ML Docs](https://rubixml.github.io/ML/latest/ "Link to resource") – Comprehensive documentation for Rubix ML
 
---

## Support this project

If this project helps you, you can support development here:

💖 [Sponsor me on GitHub](https://github.com/sponsors/apphp)

---

## License

This list is licensed under the MIT License – see LICENSE for details.

## Contributing

Contributions are welcome!  
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details, including criteria for adding new projects (maintenance, documentation, tests, etc).

[↑ Back to top](#Awesome-PHP-Machine-Learning--AI)
