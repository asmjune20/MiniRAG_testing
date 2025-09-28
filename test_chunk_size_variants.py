#!/usr/bin/env python3
"""
Chunk Size Variants Tester for MiniRAG
Tests different chunk sizes to understand their impact on performance and quality
"""

import warnings
import time
import json
import asyncio
import logging
from pathlib import Path
from log_manager import LogManager
from test_config import *
from minirag import MiniRAG, QueryParam
from minirag.utils import EmbeddingFunc

# Import the necessary LLM and embedding functions
from minirag.llm.openai import gpt_4o_mini_complete, openai_embed

# Import PDF reading capability
import pipmaster as pm
if not pm.is_installed("pypdf2"):
    pm.install("pypdf2")
from PyPDF2 import PdfReader
import hashlib

class ChunkSizeTester:
    def __init__(self, pdf_name):
        """Initialize the chunk size tester with logging setup for a specific PDF"""
        
        # Extract PDF name without extension for folder/log naming
        self.pdf_base_name = Path(pdf_name).stem
        
        # Set up logging with PDF-specific log file name
        log_file_name = f"chunk_size_test_{self.pdf_base_name}.log"
        
        # Set up logging with LogManager
        self.log_manager = LogManager(
            log_dir=LOG_DIR,
            log_file_name=log_file_name,
            log_file_size=LOG_FILE_SIZE,
            backup_count=LOG_BACKUP_COUNT,
            debug_to_primary_only=DEBUG_TO_PRIMARY_ONLY
        )
        
        # Get logger for this test
        self.logger = self.log_manager.get_logger("ChunkSizeTester")
        
        # Create response storage directory for this PDF
        self.response_dir = RESPONSE_BASE_DIR / f"{self.pdf_base_name}_chunk_variants"
        self.response_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MiniRAG instances with different chunk sizes
        self.rag_instances = {}
        
        # Track event loop to prevent multiple asyncio.run() calls
        self._event_loop = None
        
        self.logger.info("Chunk Size Tester initialized")
        self.logger.info(f"Log directory: {LOG_DIR}")
        self.logger.info(f"Log file: {log_file_name}")
        self.logger.info(f"Response storage: {self.response_dir}")
        self.logger.info(f"Chunk size variants: {CHUNK_SIZE_VARIANTS}")
        self.logger.info(f"Chunk overlap variants: {CHUNK_OVERLAP_VARIANTS}")
        
    def _read_pdf_content(self, pdf_path):
        """Extract text content from PDF file"""
        try:
            reader = PdfReader(str(pdf_path))
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            return content.strip()
        except Exception as e:
            self.logger.error(f"❌ Failed to read PDF {pdf_path.name}: {e}")
            raise
    
    async def _get_rag_instance(self, pdf_path, chunk_size, chunk_overlap):
        """Get or create a MiniRAG instance for the given PDF with specific chunk size"""
        instance_key = f"{pdf_path}_{chunk_size}_{chunk_overlap}"
        
        if instance_key not in self.rag_instances:
            self.logger.info(f"🏗️  Creating MiniRAG instance for {pdf_path.name} with chunk_size={chunk_size}, overlap={chunk_overlap}")
            
            # Create a unique working directory for each chunk size variant
            working_dir = Path(f"./test_working_dir_chunk_{chunk_size}")
            working_dir.mkdir(parents=True, exist_ok=True)
            
            # Create embedding function
            async def embed_func(texts):
                return await openai_embed(
                    texts,
                    model=EMBEDDING_MODEL,
                    api_key=EMBEDDING_API_KEY,
                    base_url="https://api.openai.com/v1",
                )
            
            embedding_func = EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=MAX_EMBED_TOKENS,
                func=embed_func
            )
            
            # Create LLM function
            async def llm_model_func(prompt, system_prompt=None, **kwargs):
                safe_max_tokens = min(MAX_TOKENS, 4000)
                return await gpt_4o_mini_complete(
                    prompt, 
                    system_prompt=system_prompt,
                    api_key=LLM_API_KEY,
                    base_url="https://api.openai.com/v1",
                    max_tokens=safe_max_tokens,
                    **kwargs
                )
            
            # Create MiniRAG instance with specific chunk size
            safe_max_tokens = min(MAX_TOKENS, 4000)
            rag = MiniRAG(
                working_dir=str(working_dir),
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                llm_model_name=LLM_MODEL,
                chunk_token_size=chunk_size,
                chunk_overlap_token_size=chunk_overlap,
                llm_model_max_token_size=safe_max_tokens,
                kv_storage="JsonKVStorage",
                graph_storage="NetworkXStorage",
                vector_storage="NanoVectorDBStorage",
                doc_status_storage="JsonDocStatusStorage",
            )
            
            # Set logger after creation
            rag.logger = self.logger
            
            # Also redirect the global MiniRAG logger
            from minirag.utils import logger as minirag_logger
            minirag_logger.handlers.clear()
            primary_handler = list(self.log_manager.handlers.values())[0]
            minirag_logger.addHandler(primary_handler)
            minirag_logger.setLevel(logging.INFO)
            
            self.rag_instances[instance_key] = rag
            self.logger.info(f"✅ MiniRAG instance created for {pdf_path.name} with chunk_size={chunk_size}")
            
            # Index the document content
            pdf_content = self._read_pdf_content(pdf_path)
            self.logger.info(f"📄 Reading PDF content: {pdf_path.name}")
            self.logger.info(f"📄 Extracted {len(pdf_content)} characters from PDF")
            self.logger.info(f"📄 Indexing document with chunk_size={chunk_size}, overlap={chunk_overlap}...")
            await rag.ainsert(pdf_content)
            self.logger.info(f"✅ Document indexed successfully with chunk_size={chunk_size}")
        
        return self.rag_instances[instance_key]
    
    def _safe_response_processing(self, response):
        """Safely process response data from MiniRAG"""
        try:
            if response is None:
                self.logger.warning("⚠️  Received None response from MiniRAG")
                return "No response generated", 0
            
            if isinstance(response, str):
                return response, len(response)
            
            if isinstance(response, (list, tuple)):
                if len(response) > 0:
                    first_item = response[0]
                    if isinstance(first_item, dict):
                        text = first_item.get('text', str(first_item))
                    else:
                        text = str(first_item)
                    return text, len(text)
                else:
                    return "Empty response", 0
            
            if isinstance(response, dict):
                text = response.get('response', response.get('text', str(response)))
                return text, len(str(text))
            
            text = str(response)
            return text, len(text)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing response: {e}")
            return f"Error processing response: {e}", 0
    
    def save_response_to_json(self, query, mode, chunk_size, chunk_overlap, response_data, response_time, pdf_path):
        """Save response data to JSON file with chunk size information"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create filename with timestamp and chunk info
        filename = f"{self.pdf_base_name}_chunk_{chunk_size}_{mode}_{timestamp}.json"
        filepath = self.response_dir / filename
        
        # Safely process response data
        response_text, response_length = self._safe_response_processing(response_data)
        
        # Prepare data to save
        response_data_to_save = {
            "metadata": {
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "query": query,
                "mode": mode,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "timestamp": timestamp,
                "response_time_seconds": response_time,
                "response_length_chars": response_length,
                "test_type": "chunk_size_variants",
                "raw_response_type": str(type(response_data))
            },
            "response": response_text,
            "raw_response": response_data
        }
        
        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data_to_save, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"💾 Response saved to: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"❌ Failed to save response: {e}")
            return None
    
    async def run_query_async(self, query, mode, chunk_size, chunk_overlap, pdf_path):
        """Run a single query asynchronously with specific chunk size"""
        self.logger.info(f"🔍 Running query in '{mode}' mode with chunk_size={chunk_size}")
        self.logger.info(f"   Query: {query}")
        self.logger.info(f"   PDF: {pdf_path.name}")
        
        # Get MiniRAG instance with specific chunk size
        rag = await self._get_rag_instance(pdf_path, chunk_size, chunk_overlap)
        
        # Create QueryParam
        safe_max_tokens = min(MAX_TOKENS, 4000)
        param = QueryParam(
            mode=mode,
            top_k=TOP_K,
            max_token_for_text_unit=safe_max_tokens,
            max_token_for_global_context=safe_max_tokens,
            max_token_for_local_context=safe_max_tokens,
        )
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the query
            response = await rag.aquery(query, param)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Safely process the response
            response_text, response_length = self._safe_response_processing(response)
            
            self.logger.info(f"✅ Query completed successfully")
            self.logger.info(f"   Response time: {response_time:.2f}s")
            self.logger.info(f"   Response length: {response_length} characters")
            self.logger.info(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
            
            # Save response to JSON file
            saved_file = self.save_response_to_json(query, mode, chunk_size, chunk_overlap, response, response_time, pdf_path)
            
            return {
                "success": True,
                "response": response_text,
                "response_time": response_time,
                "response_length": response_length,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "saved_file": saved_file,
                "raw_response": response
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"❌ Query failed: {e}")
            self.logger.error(f"   Error type: {type(e)}")
            self.logger.error(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
            return {
                "success": False,
                "error": str(e),
                "error_type": str(type(e)),
                "response_time": response_time,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
    
    def run_query(self, query, mode, chunk_size, chunk_overlap, pdf_path):
        """Synchronous wrapper for async query"""
        try:
            if self._event_loop is None or self._event_loop.is_closed():
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
            
            return self._event_loop.run_until_complete(
                self.run_query_async(query, mode, chunk_size, chunk_overlap, pdf_path)
            )
        except Exception as e:
            self.logger.error(f"❌ Event loop error: {e}")
            try:
                return asyncio.run(self.run_query_async(query, mode, chunk_size, chunk_overlap, pdf_path))
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback event loop also failed: {fallback_error}")
                return {
                    "success": False,
                    "error": f"Event loop error: {e}, fallback failed: {fallback_error}",
                    "response_time": 0,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
    
    def test_chunk_size_variants(self, query, pdf_path, modes=["light", "mini"]):
        """Test a single query across different chunk sizes and modes"""
        self.logger.info(f"🔄 Testing chunk size variants for query: {query}")
        
        results = {}
        
        for mode in modes:
            results[mode] = {}
            self.logger.info(f"📊 Testing {mode} mode across chunk sizes...")
            
            for i, chunk_size in enumerate(CHUNK_SIZE_VARIANTS):
                chunk_overlap = CHUNK_OVERLAP_VARIANTS[i] if i < len(CHUNK_OVERLAP_VARIANTS) else CHUNK_OVERLAP_VARIANTS[-1]
                
                self.logger.info(f"   Testing chunk_size={chunk_size}, overlap={chunk_overlap}")
                result = self.run_query(query, mode, chunk_size, chunk_overlap, pdf_path)
                results[mode][chunk_size] = result
                
                if result["success"]:
                    self.logger.info(f"   ✅ Success: {result['response_time']:.2f}s, {result['response_length']} chars")
                else:
                    self.logger.error(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def analyze_results(self, results):
        """Analyze and compare results across chunk sizes"""
        self.logger.info(f"📊 Analyzing results across chunk sizes...")
        
        analysis = {}
        
        for mode, mode_results in results.items():
            analysis[mode] = {
                "chunk_sizes": [],
                "response_times": [],
                "response_lengths": [],
                "success_rates": []
            }
            
            for chunk_size, result in mode_results.items():
                analysis[mode]["chunk_sizes"].append(chunk_size)
                analysis[mode]["response_times"].append(result.get("response_time", 0))
                analysis[mode]["response_lengths"].append(result.get("response_length", 0))
                analysis[mode]["success_rates"].append(1 if result.get("success", False) else 0)
            
            # Find best performing chunk size
            if analysis[mode]["response_times"]:
                best_time_idx = min(range(len(analysis[mode]["response_times"])), 
                                  key=lambda i: analysis[mode]["response_times"][i])
                best_chunk_size = analysis[mode]["chunk_sizes"][best_time_idx]
                best_time = analysis[mode]["response_times"][best_time_idx]
                
                self.logger.info(f"🏆 {mode} mode - Best chunk size: {best_chunk_size} ({best_time:.2f}s)")
        
        return analysis

def main():
    """Main test execution function"""
    print("🚀 MiniRAG Chunk Size Variants Tester")
    print("=" * 60)
    print(f"📚 Using PDF: {PDF_SELECTION}")
    print(f"🔍 Test queries: {len(TEST_QUERIES)} questions")
    print(f"📊 Chunk sizes: {CHUNK_SIZE_VARIANTS}")
    print(f"📊 Chunk overlaps: {CHUNK_OVERLAP_VARIANTS}")
    print(f"📊 Logging to: {LOG_DIR}/")
    print(f"💾 Responses stored in: {RESPONSE_BASE_DIR}/")
    print("🔬 This version tests different chunk sizes to optimize performance")
    print("=" * 60)
    
    # Get PDF files to test
    pdf_files = []
    if isinstance(PDF_SELECTION, list):
        pdf_dir = Path("./test_pdf")
        for pdf_name in PDF_SELECTION:
            pdf_path = pdf_dir / pdf_name
            if pdf_path.exists():
                pdf_files.append(pdf_path)
            else:
                print(f"⚠️  PDF not found: {pdf_name}")
    
    if not pdf_files:
        print("❌ No PDF files found for testing")
        return
    
    # Test each PDF
    for pdf_path in pdf_files:
        print(f"\n📚 Testing PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        
        # Create tester for this PDF
        tester = ChunkSizeTester(pdf_path.name)
        
        # Test each query
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n🔍 Test {i}/{len(TEST_QUERIES)}: {query}")
            
            # Test chunk size variants
            results = tester.test_chunk_size_variants(query, pdf_path)
            
            # Analyze results
            analysis = tester.analyze_results(results)
            
            # Print summary
            print(f"\n📊 Results Summary:")
            for mode in ["light", "mini"]:
                if mode in analysis:
                    print(f"   {mode.upper()} mode:")
                    for j, chunk_size in enumerate(analysis[mode]["chunk_sizes"]):
                        time_val = analysis[mode]["response_times"][j]
                        length_val = analysis[mode]["response_lengths"][j]
                        success = analysis[mode]["success_rates"][j]
                        status = "✅" if success else "❌"
                        print(f"     {status} Chunk {chunk_size}: {time_val:.2f}s, {length_val} chars")
        
        print(f"\n📋 Check the detailed logs in {LOG_DIR}/ for:")
        print(f"   🎯 Chunk content previews during processing")
        print(f"   🕸️  Entity extraction details for each chunk size")
        print(f"   🔍 Performance comparisons across chunk sizes")
        print(f"💾 Full responses saved in: {tester.response_dir}")

if __name__ == "__main__":
    main()
