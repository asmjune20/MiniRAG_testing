#!/usr/bin/env python3
"""
Local MiniRAG Light vs Mini Mode Tester with Deep Logging
This version bypasses the HTTP API to directly use MiniRAG instances with LogManager
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

class LocalMiniRAGTester:
    def __init__(self, pdf_name):
        """Initialize the local tester with logging setup for a specific PDF"""
        
        # Extract PDF name without extension for folder/log naming
        self.pdf_base_name = Path(pdf_name).stem
        
        # Set up logging with PDF-specific log file name
        log_file_name = f"miniraq_local_test_{self.pdf_base_name}.log"
        
        # Set up logging with LogManager
        self.log_manager = LogManager(
            log_dir=LOG_DIR,
            log_file_name=log_file_name,
            log_file_size=LOG_FILE_SIZE,
            backup_count=LOG_BACKUP_COUNT,
            debug_to_primary_only=DEBUG_TO_PRIMARY_ONLY
        )
        
        # Get logger for this test
        self.logger = self.log_manager.get_logger("LocalMiniRAGTester")
        
        # Create response storage directory for this PDF
        self.response_dir = RESPONSE_BASE_DIR / f"{self.pdf_base_name}_local"
        self.response_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MiniRAG instances with logger
        self.rag_instances = {}
        
        # Track event loop to prevent multiple asyncio.run() calls
        self._event_loop = None
        
        self.logger.info("Local MiniRAG Tester initialized")
        self.logger.info(f"Log directory: {LOG_DIR}")
        self.logger.info(f"Log file: {log_file_name}")
        self.logger.info(f"Response storage: {self.response_dir}")
        
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
    
    async def _get_rag_instance(self, pdf_path):
        """Get or create a MiniRAG instance for the given PDF"""
        pdf_key = str(pdf_path)
        
        if pdf_key not in self.rag_instances:
            self.logger.info(f"🏗️  Creating new MiniRAG instance for {pdf_path.name}")
            
            # Use the existing main working directory instead of creating a new one
            # This ensures we use the same data that the server version uses
            working_dir = Path("./test_working_dir")  # Use main working directory
            
            # Create embedding function
            async def embed_func(texts):
                return await openai_embed(
                    texts,
                    model=EMBEDDING_MODEL,
                    api_key=EMBEDDING_API_KEY,
                    base_url="https://api.openai.com/v1",  # Provide base_url to avoid env var lookup
                )
            
            embedding_func = EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=MAX_EMBED_TOKENS,
                func=embed_func
            )
            
            # Create LLM function
            async def llm_model_func(prompt, system_prompt=None, **kwargs):
                # Use safe token limit for gpt-4o-mini (max 16384, use 4000 for safety)
                safe_max_tokens = min(MAX_TOKENS, 4000)
                return await gpt_4o_mini_complete(
                    prompt, 
                    system_prompt=system_prompt,
                    api_key=LLM_API_KEY,
                    base_url="https://api.openai.com/v1",  # Provide base_url to avoid env var lookup
                    max_tokens=safe_max_tokens,
                    **kwargs
                )
            
            # Create MiniRAG instance with logger support
            # Use safe token limit for gpt-4o-mini
            safe_max_tokens = min(MAX_TOKENS, 4000)
            rag = MiniRAG(
                working_dir=str(working_dir),
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                llm_model_name=LLM_MODEL,
                chunk_token_size=CHUNK_SIZE,
                chunk_overlap_token_size=CHUNK_OVERLAP_SIZE,
                llm_model_max_token_size=safe_max_tokens,
                # Explicitly specify storage classes to match server
                kv_storage="JsonKVStorage",
                graph_storage="NetworkXStorage",
                vector_storage="NanoVectorDBStorage",
                doc_status_storage="JsonDocStatusStorage",
            )
            
            # Set logger after creation
            rag.logger = self.logger
            
            # Also redirect the global MiniRAG logger to use the same log file
            from minirag.utils import logger as minirag_logger
            minirag_logger.handlers.clear()  # Remove existing handlers
            # Get the primary handler from LogManager
            primary_handler = list(self.log_manager.handlers.values())[0]
            minirag_logger.addHandler(primary_handler)
            minirag_logger.setLevel(logging.INFO)
            
            self.rag_instances[pdf_key] = rag
            self.logger.info(f"✅ MiniRAG instance created for {pdf_path.name}")
            
            # Check if document is already indexed in the main working directory
            doc_status_file = working_dir / "kv_store_doc_status.json"
            if doc_status_file.exists():
                try:
                    with open(doc_status_file, 'r') as f:
                        doc_status = json.load(f)
                    
                    # Check if this PDF is already indexed
                    pdf_content = self._read_pdf_content(pdf_path)
                    pdf_hash = hashlib.md5(pdf_content.encode()).hexdigest()
                    
                    already_indexed = False
                    for doc_id, doc_info in doc_status.items():
                        if doc_info.get("content", "").strip() == pdf_content.strip():
                            already_indexed = True
                            self.logger.info(f"📄 Document already indexed in main working directory")
                            break
                    
                    if not already_indexed:
                        self.logger.info(f"📄 Document not found in main working directory, indexing...")
                        # Index the document content
                        self.logger.info(f"📄 Reading PDF content: {pdf_path.name}")
                        self.logger.info(f"📄 Extracted {len(pdf_content)} characters from PDF")
                        self.logger.info(f"📄 Indexing document content...")
                        await rag.ainsert(pdf_content)  # Use await instead of asyncio.run()
                        self.logger.info(f"✅ Document indexed successfully")
                    else:
                        self.logger.info(f"📄 Using existing indexed document")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not check existing index, proceeding with new indexing: {e}")
                    # Fallback to indexing
                    pdf_content = self._read_pdf_content(pdf_path)
                    self.logger.info(f"📄 Reading PDF content: {pdf_path.name}")
                    self.logger.info(f"📄 Extracted {len(pdf_content)} characters from PDF")
                    self.logger.info(f"📄 Indexing document content...")
                    await rag.ainsert(pdf_content)
                    self.logger.info(f"✅ Document indexed successfully")
            else:
                self.logger.info(f"📄 No existing index found, creating new index...")
                # Index the document content
                pdf_content = self._read_pdf_content(pdf_path)
                self.logger.info(f"📄 Reading PDF content: {pdf_path.name}")
                self.logger.info(f"📄 Extracted {len(pdf_content)} characters from PDF")
                self.logger.info(f"📄 Indexing document content...")
                await rag.ainsert(pdf_content)
                self.logger.info(f"✅ Document indexed successfully")
        
        return self.rag_instances[pdf_key]
    
    def _safe_response_processing(self, response):
        """Safely process response data from MiniRAG"""
        try:
            if response is None:
                self.logger.warning("⚠️  Received None response from MiniRAG")
                return "No response generated", 0
            
            if isinstance(response, str):
                return response, len(response)
            
            if isinstance(response, (list, tuple)):
                # Handle tuple/list responses (common in mini mode)
                if len(response) > 0:
                    first_item = response[0]
                    if isinstance(first_item, dict):
                        # Extract text from dictionary if possible
                        text = first_item.get('text', str(first_item))
                    else:
                        text = str(first_item)
                    return text, len(text)
                else:
                    return "Empty response", 0
            
            if isinstance(response, dict):
                # Handle dictionary responses
                text = response.get('response', response.get('text', str(response)))
                return text, len(str(text))
            
            # Fallback for any other type
            text = str(response)
            return text, len(text)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing response: {e}")
            return f"Error processing response: {e}", 0
    
    def save_response_to_json(self, query, mode, response_data, response_time, pdf_path):
        """Save response data to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create filename with timestamp
        filename = f"{self.pdf_base_name}_local_{mode}_{timestamp}.json"
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
                "timestamp": timestamp,
                "response_time_seconds": response_time,
                "response_length_chars": response_length,
                "test_type": "local_direct",
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
    
    async def run_query_async(self, query, mode, pdf_path):
        """Run a single query asynchronously in the specified mode"""
        self.logger.info(f"🔍 Running query in '{mode}' mode")
        self.logger.info(f"   Query: {query}")
        self.logger.info(f"   PDF: {pdf_path.name}")
        
        # Get MiniRAG instance
        rag = await self._get_rag_instance(pdf_path)
        
        # Create QueryParam
        # Use safe token limits for gpt-4o-mini
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
            # Run the query directly with the MiniRAG instance
            # This will trigger our detailed logging in operate.py
            response = await rag.aquery(query, param)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Safely process the response
            response_text, response_length = self._safe_response_processing(response)
            
            self.logger.info(f"✅ Query completed successfully")
            self.logger.info(f"   Response time: {response_time:.2f}s")
            self.logger.info(f"   Response length: {response_length} characters")
            self.logger.info(f"   Response type: {type(response)}")
            
            # Check performance thresholds
            if response_time > EXPECTED_RESPONSE_TIME:
                self.logger.warning(f"⚠️  Response time ({response_time:.2f}s) exceeded threshold ({EXPECTED_RESPONSE_TIME}s)")
            
            if response_length < MIN_RESPONSE_LENGTH:
                self.logger.warning(f"⚠️  Response length ({response_length} chars) below minimum ({MIN_RESPONSE_LENGTH} chars)")
            
            # Save response to JSON file
            saved_file = self.save_response_to_json(query, mode, response, response_time, pdf_path)
            
            return {
                "success": True,
                "response": response_text,
                "response_time": response_time,
                "response_length": response_length,
                "saved_file": saved_file,
                "raw_response": response
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"❌ Query failed: {e}")
            self.logger.error(f"   Error type: {type(e)}")
            self.logger.error(f"   Error details: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": str(type(e)),
                "response_time": response_time
            }
    
    def run_query(self, query, mode, pdf_path):
        """Synchronous wrapper for async query"""
        # Use existing event loop if available, otherwise create new one
        try:
            if self._event_loop is None or self._event_loop.is_closed():
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
            
            return self._event_loop.run_until_complete(
                self.run_query_async(query, mode, pdf_path)
            )
        except Exception as e:
            self.logger.error(f"❌ Event loop error: {e}")
            # Fallback to creating a new event loop
            try:
                return asyncio.run(self.run_query_async(query, mode, pdf_path))
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback event loop also failed: {fallback_error}")
                return {
                    "success": False,
                    "error": f"Event loop error: {e}, fallback failed: {fallback_error}",
                    "response_time": 0
                }
    
    def compare_modes(self, query, pdf_path):
        """Compare 'light' and 'mini' modes for a single query"""
        self.logger.info(f"🔄 Comparing modes for query: {query}")
        
        # Test both modes
        light_result = self.run_query(query, "light", pdf_path)
        mini_result = self.run_query(query, "mini", pdf_path)
        
        # Compare results
        if light_result["success"] and mini_result["success"]:
            time_diff = light_result["response_time"] - mini_result["response_time"]
            length_diff = light_result["response_length"] - mini_result["response_length"]
            
            self.logger.info(f"📊 Comparison Results:")
            self.logger.info(f"   Light mode: {light_result['response_time']:.2f}s, {light_result['response_length']} chars")
            self.logger.info(f"   Mini mode: {mini_result['response_time']:.2f}s, {mini_result['response_length']} chars")
            self.logger.info(f"   Time difference: {time_diff:+.2f}s")
            self.logger.info(f"   Length difference: {length_diff:+d} chars")
            
            # Determine winner
            if time_diff > 0:
                self.logger.info(f"🏆 Mini mode is {time_diff:.2f}s faster")
            else:
                self.logger.info(f"🏆 Light mode is {abs(time_diff):.2f}s faster")
                
            if length_diff > 0:
                self.logger.info(f"📝 Light mode provides {length_diff} more characters")
            else:
                self.logger.info(f"📝 Mini mode provides {abs(length_diff)} more characters")
        else:
            # Log failures for debugging
            if not light_result["success"]:
                self.logger.error(f"❌ Light mode failed: {light_result.get('error', 'Unknown error')}")
            if not mini_result["success"]:
                self.logger.error(f"❌ Mini mode failed: {mini_result.get('error', 'Unknown error')}")
        
        return light_result, mini_result

def main():
    """Main test execution function"""
    print("🚀 Local MiniRAG Light vs Mini Mode Tester with Deep Logging")
    print("=" * 60)
    print(f"📚 Using PDF: {PDF_SELECTION}")
    print(f"🔍 Test queries: {len(TEST_QUERIES)} questions")
    print(f"📊 Logging to: {LOG_DIR}/")
    print(f"💾 Responses stored in: {RESPONSE_BASE_DIR}/")
    print("🔬 This version provides detailed entity extraction and graph traversal logs")
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
        tester = LocalMiniRAGTester(pdf_path.name)
        
        # Test each query
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n🔍 Test {i}/{len(TEST_QUERIES)}: {query}")
            
            # Compare modes
            light_result, mini_result = tester.compare_modes(query, pdf_path)
            
            # Print summary
            if light_result["success"] and mini_result["success"]:
                time_diff = light_result["response_time"] - mini_result["response_time"]
                print(f"   ⚡ Light: {light_result['response_time']:.2f}s, {light_result['response_length']} chars")
                print(f"   🚀 Mini: {mini_result['response_time']:.2f}s, {mini_result['response_length']} chars")
                print(f"   📊 Difference: {time_diff:+.2f}s, {light_result['response_length'] - mini_result['response_length']:+d} chars")
            else:
                # Show error information
                if not light_result["success"]:
                    print(f"   ❌ Light mode failed: {light_result.get('error', 'Unknown error')}")
                if not mini_result["success"]:
                    print(f"   ❌ Mini mode failed: {mini_result.get('error', 'Unknown error')}")
        
        print(f"\n📋 Check the detailed logs in {LOG_DIR}/ for:")
        print(f"   🎯 Entity extraction details")
        print(f"   🕸️  Graph traversal paths")
        print(f"   🔍 Keyword extraction process")
        print(f"   🌐 Context building steps")
        print(f"💾 Full responses saved in: {tester.response_dir}")

if __name__ == "__main__":
    main() 