# MiniRAG_fresh: Enhanced MiniRAG with EigenAI Integration

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)

## 🌟 Overview

**MiniRAG_fresh** is an enhanced version of MiniRAG with comprehensive EigenAI integration and advanced testing capabilities. This repository extends the original MiniRAG framework with specialized support for EigenAI models and includes a robust testing framework for performance evaluation.

## 🚀 Key Features

### 🔧 **EigenAI Integration**
- **Multiple Model Support**: GPT-OSS 120B, Llama 3.2 1B, Gemma 3 27B, Llama 3.1 8B
- **Token Limit Optimization**: Handles EigenAI's 1,964 token input limit with 600-token chunk strategy
- **Custom Entity Extraction**: Specialized JSON response handling for EigenAI models
- **Robust Error Handling**: Comprehensive retry logic and error management

### 🧪 **Advanced Testing Framework**
- **Performance Comparison**: Light vs Mini mode benchmarking
- **Chunk Size Optimization**: Testing with 400, 800, and 1200 token variants
- **Real-world Testing**: PDF processing with actual documents
- **Detailed Logging**: Comprehensive test tracking and analysis

### 📊 **Multi-Model Support**
- **EigenAI Models**: Full integration with EigenAI's model suite
- **OpenAI Models**: GPT-4o-mini and other OpenAI models
- **HuggingFace Models**: Local and cloud-based HF models
- **Extensible Architecture**: Easy addition of new LLM providers

## 🏗️ Architecture

```
MiniRAG_fresh/
├── minirag/                    # Core MiniRAG framework
│   ├── llm/                   # LLM integrations
│   │   ├── eigenai.py         # EigenAI implementation
│   │   ├── openai.py          # OpenAI implementation
│   │   └── ...                # Other LLM providers
│   ├── kg/                    # Knowledge graph implementations
│   ├── operate.py             # Core operations
│   ├── operate_eigenai.py     # EigenAI-specific operations
│   └── ...
├── test_*.py                  # Comprehensive testing suite
├── log_manager.py             # Advanced logging system
└── test_pdf/                  # Test documents
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

### Basic Usage

```python
import asyncio
from minirag import MiniRAG
from minirag.llm.eigenai import eigenai_gpt_oss_120b_complete

async def main():
    # Initialize MiniRAG with EigenAI
    rag = MiniRAG(
        working_dir="./test_working_dir",
        chunk_token_size=600,  # Optimized for EigenAI token limits
        chunk_overlap_token_size=50,
        llm_model_func=eigenai_gpt_oss_120b_complete,
        llm_model_name="gpt-oss-120b-f16"
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

# Test different chunk sizes
python test_chunk_size_variants.py

# Preview chunk analysis
python test_chunk_preview.py
```

### EigenAI-Specific Testing

```bash
# Test EigenAI integration
python -c "from minirag.llm.eigenai import test_eigenai; import asyncio; asyncio.run(test_eigenai())"
```

## 🔧 Configuration

### EigenAI Setup

```python
# EigenAI configuration
EIGENAI_BASE_URL = "http://192.222.58.254:8001"
EIGENAI_API_KEY = "your-api-key"
EIGENAI_MODEL = "gpt-oss-120b-f16"

# Token limits (optimized for EigenAI)
MAX_INPUT_TOKENS = 1900  # Input prompt limit
MAX_TOKENS = 4000        # Response generation limit
CHUNK_SIZE = 600         # Optimized chunk size
```

### Testing Configuration

```python
# Test configuration
TEST_QUERIES = [
    "What were some of Kalpana Chawla's hobbies?",
    "Explain about Kalpana Chawla's husband",
    "Explain Kalpana Chawla's educational background",
    "When did Kalpana Chawla start working at NASA?"
]

CHUNK_SIZE_VARIANTS = [400, 800, 1200]
```

## 📊 Performance Results

### EigenAI Model Performance
- **GPT-OSS 120B**: Primary model with excellent performance
- **Llama 3.2 1B**: Lightweight alternative
- **Gemma 3 27B**: Balanced performance and efficiency
- **Llama 3.1 8B**: Mid-range option

### Chunk Size Optimization
- **600 tokens**: Optimal for EigenAI token limits
- **400 tokens**: Conservative approach for complex prompts
- **800 tokens**: Higher context but may hit limits
- **1200 tokens**: Maximum context (may require prompt optimization)

## 🛠️ Development

### Adding New LLM Providers

```python
# Example: Adding a new LLM provider
async def custom_llm_complete(prompt: str, **kwargs) -> str:
    # Your implementation here
    pass

# Use with MiniRAG
rag = MiniRAG(
    llm_model_func=custom_llm_complete,
    llm_model_name="custom-model"
)
```

### Custom Entity Extraction

```python
# Example: Custom entity extraction for specific models
from minirag.operate_eigenai import extract_entities_eigenai

# Use EigenAI-specific extraction
entities = await extract_entities_eigenai(
    chunks=chunks,
    knowledge_graph_inst=kg,
    entity_vdb=entity_vdb,
    global_config=config
)
```

## 📁 Project Structure

```
MiniRAG_fresh/
├── minirag/                    # Core framework
│   ├── llm/                   # LLM integrations
│   │   ├── eigenai.py         # EigenAI models
│   │   ├── openai.py          # OpenAI models
│   │   └── ...                # Other providers
│   ├── kg/                    # Knowledge graph backends
│   ├── operate.py             # Core operations
│   ├── operate_eigenai.py     # EigenAI operations
│   └── ...
├── test_light_vs_mini_local.py    # Main testing script
├── test_chunk_size_variants.py    # Chunk size testing
├── test_chunk_preview.py          # Chunk analysis
├── test_config.py                 # Test configuration
├── log_manager.py                 # Logging system
├── test_pdf/                      # Test documents
└── dataset/                       # Benchmark datasets
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
- **EigenAI**: For providing access to their model suite
- **LightRAG**: For the foundational RAG framework

## 📚 References

- [MiniRAG Paper](https://arxiv.org/abs/2501.06713)
- [EigenAI Documentation](https://eigenai.com)
- [LightRAG Repository](https://github.com/HKUDS/LightRAG)

## 🔗 Links

- **Repository**: [https://github.com/asmjune20/MiniRAG_testing](https://github.com/asmjune20/MiniRAG_testing)
- **Issues**: [Report bugs or request features](https://github.com/asmjune20/MiniRAG_testing/issues)
- **Discussions**: [Join the conversation](https://github.com/asmjune20/MiniRAG_testing/discussions)

---

**Made with ❤️ by [June Cho](https://github.com/asmjune20)**