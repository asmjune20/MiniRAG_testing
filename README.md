# MiniRAG_fresh: Enhanced MiniRAG Testing Framework

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)

## 🌟 Overview

**MiniRAG_fresh** is an enhanced MiniRAG framework with comprehensive testing capabilities, optimized for **GPT-4o-mini** performance evaluation. This repository provides advanced testing tools for comparing Light vs Mini modes, chunk size optimization, and detailed performance analysis.

## 🚀 Key Features

- **🧪 Performance Testing**: Light vs Mini mode benchmarking with GPT-4o-mini
- **📊 Chunk Size Optimization**: Testing with 400, 800, and 1200 token variants
- **📄 Real-world Testing**: PDF processing with actual documents
- **📝 Advanced Logging**: Comprehensive test tracking with LogManager
- **🔧 Multi-Model Support**: GPT-4o-mini (primary), EigenAI (experimental), OpenAI suite

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/asmjune20/MiniRAG_testing.git
cd MiniRAG_testing
pip install -e .
```

### Environment Setup

Create a `.env` file:

```bash
LLM_BINDING_API_KEY=your_openai_api_key_here
EMBEDDING_BINDING_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### Basic Usage

```python
import asyncio
from minirag import MiniRAG
from minirag.llm.openai import gpt_4o_mini_complete

async def main():
    rag = MiniRAG(
        working_dir="./test_working_dir",
        chunk_token_size=1200,
        llm_model_func=gpt_4o_mini_complete,
        llm_model_name="gpt-4o-mini"
    )
    
    await rag.ainsert("Your document text here...")
    response = await rag.aquery("Your question here...")
    print(response)

asyncio.run(main())
```

## 🧪 Testing Framework

### Run Tests

```bash
# Light vs Mini mode comparison
python test_light_vs_mini_local.py

# Chunk size optimization (400, 800, 1200 tokens)
python test_chunk_size_variants.py

# Chunk analysis and preview
python test_chunk_preview.py
```

### Test Configuration

Edit `test_config.py` to customize:

```python
TEST_QUERIES = [
    "What were some of Kalpana Chawla's hobbies?",
    "Explain about Kalpana Chawla's husband",
    # Add your questions here
]

CHUNK_SIZE_VARIANTS = [400, 800, 1200]
```

## 📊 Performance Results

| Mode | Response Time | Context Quality | Use Case |
|------|---------------|-----------------|----------|
| **Light** | 2-4 seconds | Keyword-based | Fast, simple queries |
| **Mini** | 4-6 seconds | Graph-based | Complex, detailed queries |

### Chunk Size Analysis
- **400 tokens**: Fast processing, may lose context
- **800 tokens**: Balanced performance and context  
- **1200 tokens**: Rich context, optimal for complex queries

## 🏗️ Project Structure

```
MiniRAG_fresh/
├── minirag/                    # Core MiniRAG framework
├── test_light_vs_mini_local.py    # Main performance testing
├── test_chunk_size_variants.py    # Chunk size optimization
├── test_config.py                 # Test configuration
├── log_manager.py                 # Advanced logging
└── test_pdf/                      # Test documents
```

## 🔧 Configuration

### GPT-4o-mini Settings

```python
MAX_TOKENS = 4000          # Response generation limit
CHUNK_SIZE = 1200          # Optimal chunk size
CHUNK_OVERLAP = 100        # Context overlap
EMBEDDING_DIM = 1536       # OpenAI embedding dimension
```

### Performance Thresholds

```python
EXPECTED_RESPONSE_TIME = 5.0    # seconds
MIN_RESPONSE_LENGTH = 100       # characters
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original MiniRAG**: [Tianyu Fan et al.](https://arxiv.org/abs/2501.06713)
- **OpenAI**: GPT-4o-mini and embedding models
- **LightRAG**: Foundational RAG framework

## 🔗 Links

- **Repository**: [https://github.com/asmjune20/MiniRAG_testing](https://github.com/asmjune20/MiniRAG_testing)
- **Issues**: [Report bugs or request features](https://github.com/asmjune20/MiniRAG_testing/issues)

---

**Made with ❤️ by [June Cho](https://github.com/asmjune20)**