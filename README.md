# Awesome PHP Machine Learning & AI

![Awesome](https://awesome.re/badge.svg)
![GitHub stars](https://img.shields.io/github/stars/apphp/awesome-php-ml?style=social)
![Last commit](https://img.shields.io/github/last-commit/apphp/awesome-php-ml)
![License](https://img.shields.io/github/license/apphp/awesome-php-ml)
![Link Check](https://github.com/apphp/awesome-php-ml/actions/workflows/link-check.yml/badge.svg)

The most comprehensive curated list of **Machine Learning, Artificial Intelligence, NLP, LLM and Data Science libraries for PHP**.

Inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and the broader **Awesome** ecosystem.

> **Goal:** make it easy to build intelligent systems with PHP — from classic ML to modern LLM-based workflows.

---

## Contents

- [Machine Learning](#machine-learning)
- [Deep Learning & Neural Networks](#deep-learning--neural-networks)
- [Natural Language Processing](#natural-language-processing)
- [Computer Vision](#computer-vision)
- [Math, Statistics & Linear Algebra](#math-statistics--linear-algebra)
- [LLMs & AI APIs](#llms--ai-apis)
- [Embeddings & Vector Search](#embeddings--vector-search)
- [Data Processing](#data-processing)
- [Interop & Model Serving](#interop--model-serving)
- [Tools & Utilities](#tools--utilities)
- [Resources](#resources)

---

## Machine Learning

- [dr-que/polynomial-regression](https://packagist.org/packages/dr-que/polynomial-regression) – Polynomial regression for PHP
- [php-ai/php-ml](https://gitlab.com/php-ai/php-ml) – Core machine learning algorithms for PHP
- [php-ai/php-ml-examples](https://github.com/php-ai/php-ml-examples) – Practical examples for PHP-ML
- [RubixML/RubixML](https://github.com/RubixML/RubixML) – High-level ML framework with pipelines and datasets

---

## Deep Learning & Neural Networks

- [RubixML/ML](https://github.com/RubixML/ML) – Neural networks and advanced learners
- [RubixML/Tensor](https://github.com/RubixML/Tensor) – N-dimensional tensors for numerical computing

---

## Natural Language Processing

- [angeloskath/php-nlp-tools](https://github.com/angeloskath/php-nlp-tools) – Natural Language Processing tools
- [CodeWithKyrian/transformers-php](https://github.com/CodeWithKyrian/transformers-php) – Hugging Face–style Transformer inference in PHP using ONNX
- [friteuseb/nlp_tools](https://github.com/friteuseb/nlp_tools) – Extension for nlp methods and text analysis
- [patrickschur/language-detection](https://github.com/patrickschur/language-detection) – Language detection library
- [voku/stop-words](https://github.com/voku/stop-words) – Stop word lists for many languages
- [yooper/php-text-analysis](https://github.com/yooper/php-text-analysis) – Sentiment analysis and NLP tools

---

## Computer Vision

- [Intervention/image](https://github.com/Intervention/image) – Image manipulation (useful for CV preprocessing)
- [jcupitt/vips](https://github.com/jcupitt/libvips) – Fast image processing library (PHP bindings available)
- [php-opencv/php-opencv](https://github.com/php-opencv/php-opencv) – OpenCV bindings for PHP

---

## Math, Statistics & Linear Algebra

- [markrogoyski/math-php](https://github.com/markrogoyski/math-php) – Linear algebra, statistics, and calculus
- [mcordingley/LinearAlgebra](https://github.com/mcordingley/LinearAlgebra) – Stand-alone Linear Algebra Library for PHP
- [NumPHP/NumPHP](https://github.com/NumPHP/NumPHP) – Mathematical PHP library for scientific computing
- [rindow/rindow-math-matrix](https://github.com/rindow/rindow-math-matrix) – The fundamental package for scientific matrix operation
- [RubixML/numpower](https://github.com/RubixML/numpower) – High-performance numerical computing library for PHP inspired by NumPy
- [RubixML/Tensor](https://github.com/RubixML/Tensor) – Vectorized numerical operations

---

## LLMs & AI APIs

- [google-gemini-php/client](https://github.com/google-gemini-php/client) – Google Gemini API client
- [openai-php/client](https://github.com/openai-php/client) – Official OpenAI PHP client
- [orhanerday/open-ai](https://github.com/orhanerday/open-ai) – Popular OpenAI API wrapper

### Tokenizers & Prompt Utilities

- [Gioni06/GPT3Tokenizer](https://github.com/Gioni06/GPT3Tokenizer) – PHP tokenizer compatible with GPT-3 style models
- [yethee/tiktoken-php](https://github.com/yethee/tiktoken-php) – PHP implementation of OpenAI’s *tiktoken* tokenizer for token counting and optimization

---

## Embeddings & Vector Search

- [algolia/algoliasearch-client-php](https://github.com/algolia/algoliasearch-client-php) – Algolia PHP client
- [meilisearch/meilisearch-php](https://github.com/meilisearch/meilisearch-php) – PHP client for Meilisearch
- [openai-php/laravel](https://github.com/openai-php/laravel) – Laravel OpenAI integration
- [pgvector/pgvector](https://github.com/pgvector/pgvector) – Open-source vector similarity search for Postgres
- [pgvector/pgvector-php](https://github.com/pgvector/pgvector-php) – pgVector support for PHP
- [voyanara/milvus-php-sdk](https://github.com/voyanara/milvus-php-sdk) – PHP SDK for Milvus vector database API v2

---

## Data Processing

- [flow-php/flow](https://github.com/flow-php/flow) – Data processing and ETL framework for PHP with typed pipelines
- [league/csv](https://github.com/thephpleague/csv) – CSV data processing
- [spatie/data-transfer-object](https://github.com/spatie/data-transfer-object) – Strongly typed DTOs
- [symfony/serializer](https://github.com/symfony/serializer) – Data normalization & serialization

---

## Interop & Model Serving

- [ankane/onnxruntime-php](https://github.com/ankane/onnxruntime-php) – Run ONNX models from PHP
- [FFI](https://www.php.net/manual/en/book.ffi.php) – Native C/C++ bindings for ML inference
- [grpc/grpc-php](https://github.com/grpc/grpc-php) – gRPC client for model services
- [neuron-core/neuron-ai](https://github.com/neuron-core/neuron-ai) – PHP agentic AI framework for building and orchestrating LLMs, RAG etc.

---

## Tools & Utilities

- [apphp/pretty-print](https://github.com/apphp/pretty-print) – Pretty-print PHP arrays and numeric data for ML debugging
- [nunomaduro/collision](https://github.com/nunomaduro/collision) – CLI error handling (useful for ML tools)
- [psr/log](https://github.com/php-fig/log) – Logging standard
- [spatie/laravel-data](https://github.com/spatie/laravel-data) – Typed DTOs for API & AI responses
- [swoole/phpy](https://github.com/swoole/phpy) – Bridge for calling Python from PHP via a runtime bridge
- [symfony/console](https://github.com/symfony/console) – CLI applications
- [symfony/http-client](https://github.com/symfony/http-client) – Robust HTTP client for AI APIs

---

## Laravel & Framework Integrations

- [laravel/scout](https://github.com/laravel/scout) – Search abstraction (useful for vector search)
- [openai-php/laravel](https://github.com/openai-php/laravel) – Official OpenAI Laravel integration
- [prism-php/prism](https://github.com/prism-php/prism) – Laravel interface for working with LLMs and AI providers
- [shawnveltman/laravel-openai](https://github.com/shawnveltman/laravel-openai) – Laravel wrapper for OpenAI
- [spatie/laravel-data](https://github.com/spatie/laravel-data) – Typed DTOs for AI responses
- [teamtnt/laravel-scout-tntsearch-driver](https://github.com/teamtnt/laravel-scout-tntsearch-driver) – Local full-text search

---

## Symfony & Framework Integrations
- [openai-php/symfony](https://github.com/openai-php/symfony) – OpenAI PHP for Symfony integration
    
---

## Resources

- [Awesome PHP](https://github.com/ziadoz/awesome-php)
- [ONNX Runtime](https://onnxruntime.ai)

## Learning

- [Artificial Intelligence with PHP (GitBook)](https://apphp.gitbook.io/artificial-intelligence-with-php/) – Guide and reference for doing AI/ML with PHP
- [AI для PHP-разработчиков — интуитивно и на практике (GitBook)](https://apphp.gitbook.io/ai-dlya-php-razrabotchikov-intuitivno-i-na-praktike/) – Russian guide on AI with PHP
- [Build Your Own LLM in PHP (GitBook)](https://apphp.gitbook.io/build-your-own-llm-in-php/) – Guide to building an LLM from scratch in PHP

---

## Contributing

Contributions are welcome!  
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.
