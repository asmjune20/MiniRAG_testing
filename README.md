# MiniRAG_fresh: Enhanced MiniRAG with Advanced Testing Framework

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)

## 🌟 Overview

**MiniRAG_fresh** is an enhanced version of MiniRAG with a comprehensive testing framework focused on **GPT-4o-mini performance evaluation**. This repository provides advanced testing capabilities for comparing Light vs Mini modes, chunk size optimization, and detailed performance analysis.

## 🚀 Key Features

### 🧪 **Advanced Testing Framework**
- **Performance Comparison**: Light vs Mini mode benchmarking with GPT-4o-mini
- **Chunk Size Optimization**: Testing with 400, 800, and 1200 token variants
- **Real-world Testing**: PDF processing with actual documents (Kalpana Chawla NASA biography)
- **Detailed Logging**: Comprehensive test tracking and analysis with LogManager
- **Response Quality Analysis**: Character count, response time, and accuracy metrics

### 🔧 **GPT-4o-mini Integration**
- **Optimized Configuration**: Safe token limits (4000 tokens) for reliable performance
- **OpenAI Embeddings**: text-embedding-3-small for vector similarity
- **Error Handling**: Robust retry logic and timeout management
- **Cost Optimization**: Efficient prompt engineering for minimal API calls

### 📊 **Multi-Model Support**
- **Primary**: GPT-4o-mini (main focus)
- **EigenAI**: Available for experimental use (gpt-oss-120b-f16, llama models)
- **OpenAI**: Full OpenAI model suite support
- **HuggingFace**: Local and cloud-based HF models
- **Extensible Architecture**: Easy addition of new LLM providers

## 🏗️ Architecture

```
MiniRAG_fresh/
├── minirag/                    # Core MiniRAG framework
│   ├── llm/                   # LLM integrations
│   │   ├── openai.py          # OpenAI implementation (primary)
│   │   ├── eigenai.py         # EigenAI implementation (experimental)
│   │   └── ...                # Other LLM providers
│   ├── kg/                    # Knowledge graph implementations
│   ├── operate.py             # Core operations
│   └── ...
├── test_light_vs_mini_local.py    # Main Light vs Mini comparison
├── test_chunk_size_variants.py    # Chunk size optimization testing
├── test_chunk_preview.py          # Chunk analysis and preview
├── test_config.py                 # Centralized test configuration
├── log_manager.py                 # Advanced logging system
└── test_pdf/                      # Test documents
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/asmjune20/MiniRAG_testing.git
cd MiniRAG_testing

# Install dependencies
pip install -e .
```

### Environment Setup

Create a `.env` file with your OpenAI API keys:

```bash
# OpenAI Configuration
LLM_BINDING_API_KEY=your_openai_api_key_here
EMBEDDING_BINDING_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
LLM_BINDING=openai
EMBEDDING_BINDING=openai

# RAG Configuration
TOP_K=60
COSINE_THRESHOLD=0.4
MAX_TOKENS=4000
EMBEDDING_DIM=1536
MAX_EMBED_TOKENS=1000
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
```

### Basic Usage

```python
import asyncio
from minirag import MiniRAG
from minirag.llm.openai import gpt_4o_mini_complete, openai_embed
from minirag.utils import EmbeddingFunc

async def main():
    # Initialize MiniRAG with GPT-4o-mini
    rag = MiniRAG(
        working_dir="./test_working_dir",
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        llm_model_func=gpt_4o_mini_complete,
        llm_model_name="gpt-4o-mini"
    )
    
    # Insert document
    await rag.ainsert("Your document text here...")
    
    # Query
    response = await rag.aquery("Your question here...")
    print(response)

asyncio.run(main())
```

## 🧪 Testing Framework

### Performance Testing

```bash
# Run Light vs Mini mode comparison
python test_light_vs_mini_local.py

# Test different chunk sizes (400, 800, 1200 tokens)
python test_chunk_size_variants.py

# Preview and analyze chunks
python test_chunk_preview.py
```

### Test Configuration

The testing framework uses `test_config.py` for centralized configuration:

```python
# Test queries
TEST_QUERIES = [
    "What were some of Kalpana Chawla's hobbies?",
    "Explain about Kalpana Chawla's husband",
    "Explain Kalpana Chawla's educational background",
    "When did Kalpana Chawla start working at NASA?"
]

# Chunk size variants for testing
CHUNK_SIZE_VARIANTS = [400, 800, 1200]
CHUNK_OVERLAP_VARIANTS = [50, 100, 150]

# Performance thresholds
EXPECTED_RESPONSE_TIME = 5.0  # seconds
MIN_RESPONSE_LENGTH = 100     # characters
```

## 📊 Performance Results

### GPT-4o-mini Performance
- **Response Time**: Typically 2-6 seconds for complex queries
- **Token Efficiency**: Optimized for 4000 token limit
- **Accuracy**: High-quality responses with proper context
- **Cost**: Cost-effective for production use

### Chunk Size Analysis
- **400 tokens**: Fast processing, may lose context
- **800 tokens**: Balanced performance and context
- **1200 tokens**: Rich context, optimal for complex queries
- **Overlap**: 50-150 tokens for better context continuity

### Light vs Mini Mode Comparison
- **Light Mode**: Faster, keyword-based retrieval
- **Mini Mode**: More sophisticated, graph-based retrieval
- **Performance**: Detailed metrics and response quality analysis

## 🔧 Configuration

### GPT-4o-mini Optimization

```python
# Safe token limits for GPT-4o-mini
MAX_TOKENS = 4000  # Response generation limit
MAX_EMBED_TOKENS = 1000  # Embedding token limit
CHUNK_SIZE = 1200  # Optimal chunk size
CHUNK_OVERLAP = 100  # Context overlap

# OpenAI configuration
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
```

### Testing Configuration

```python
# Performance testing
EXPECTED_RESPONSE_TIME = 5.0
MIN_RESPONSE_LENGTH = 100
LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Test documents
PDF_SELECTION = ["kalpana_chawla_nasa.pdf"]
```

## 🛠️ Development

### Adding New Test Cases

```python
# Add new test queries
TEST_QUERIES.append("Your new question here")

# Add new PDFs for testing
PDF_SELECTION.append("your_document.pdf")
```

### Custom Logging

```python
from log_manager import LogManager

# Initialize logging
log_manager = LogManager(
    log_dir="./test_logs",
    log_file_name="custom_test.log",
    log_file_size=10 * 1024 * 1024,
    backup_count=5
)

logger = log_manager.get_logger("CustomTest")
```

## 📁 Project Structure

```
MiniRAG_fresh/
├── minirag/                           # Core MiniRAG framework
│   ├── llm/                          # LLM integrations
│   │   ├── openai.py                 # OpenAI models (primary)
│   │   ├── eigenai.py                # EigenAI models (experimental)
│   │   └── ...                       # Other providers
│   ├── kg/                           # Knowledge graph backends
│   ├── operate.py                    # Core operations
│   └── ...
├── test_light_vs_mini_local.py       # Main performance testing
├── test_chunk_size_variants.py       # Chunk size optimization
├── test_chunk_preview.py             # Chunk analysis
├── test_config.py                    # Test configuration
├── log_manager.py                    # Advanced logging system
├── test_pdf/                         # Test documents
│   └── kalpana_chawla_nasa.pdf      # NASA biography for testing
└── dataset/                          # Benchmark datasets
    └── LiHua-World/                  # LiHua-World dataset
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original MiniRAG**: Based on the excellent work by [Tianyu Fan et al.](https://arxiv.org/abs/2501.06713)
- **OpenAI**: For providing GPT-4o-mini and embedding models
- **LightRAG**: For the foundational RAG framework

## 📚 References

- [MiniRAG Paper](https://arxiv.org/abs/2501.06713)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LightRAG Repository](https://github.com/HKUDS/LightRAG)

## 🔗 Links

- **Repository**: [https://github.com/asmjune20/MiniRAG_testing](https://github.com/asmjune20/MiniRAG_testing)
- **Issues**: [Report bugs or request features](https://github.com/asmjune20/MiniRAG_testing/issues)
- **Discussions**: [Join the conversation](https://github.com/asmjune20/MiniRAG_testing/discussions)

---

**Made with ❤️ by [June Cho](https://github.com/asmjune20)**