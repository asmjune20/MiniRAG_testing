"""
Modified operate.py functions for EigenAI JSON responses
"""

import json
import re
from collections import defaultdict
from typing import Union
from minirag.base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage, TextChunkSchema
from minirag.prompt import PROMPTS
from minirag.utils import clean_str, locate_json_string_body_from_string
import json_repair


async def _handle_json_entity_extraction(
    json_data: dict,
    chunk_key: str,
):
    """Handle entity extraction from JSON response"""
    entities = []
    if "entities" in json_data:
        for entity in json_data["entities"]:
            if isinstance(entity, dict) and "name" in entity and "type" in entity:
                entity_name = clean_str(entity["name"].upper())
                if not entity_name.strip():
                    continue
                entity_type = clean_str(entity["type"].upper())
                entity_description = clean_str(entity.get("description", ""))
                
                entities.append(dict(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    description=entity_description,
                    source_id=chunk_key,
                ))
    return entities


async def _handle_json_relationship_extraction(
    json_data: dict,
    chunk_key: str,
):
    """Handle relationship extraction from JSON response"""
    relationships = []
    if "relationships" in json_data:
        for rel in json_data["relationships"]:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                source_entity = clean_str(rel["source"].upper())
                target_entity = clean_str(rel["target"].upper())
                relationship_description = clean_str(rel.get("description", ""))
                relationship_strength = rel.get("strength", 5)
                
                relationships.append(dict(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    relationship_description=relationship_description,
                    relationship_strength=relationship_strength,
                    source_id=chunk_key,
                ))
    return relationships


async def extract_entities_eigenai(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """Modified entity extraction for EigenAI JSON responses"""
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["entity_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        
        # Format the prompt with the content
        hint_prompt = entity_extract_prompt.format(input_text=content)
        final_result = await use_llm_func(hint_prompt)

        # Try to parse JSON response
        try:
            # Extract JSON from the response
            json_text = locate_json_string_body_from_string(final_result)
            if json_text is None:
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', final_result, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    print(f"Could not find JSON in response: {final_result[:200]}...")
                    return

            # Parse JSON
            json_data = json_repair.loads(json_text)
            
            # Extract entities
            entities = await _handle_json_entity_extraction(json_data, chunk_key)
            for entity in entities:
                entity_name = entity["entity_name"]
                entity_type = entity["entity_type"]
                entity_description = entity["description"]
                entity_source_id = entity["source_id"]
                
                # Add to knowledge graph
                knowledge_graph_inst.add_node(
                    entity_name,
                    entity_type=entity_type,
                    description=entity_description,
                    source_id=entity_source_id,
                )
                
                # Add to vector databases
                entity_vdb.insert([entity_name], [entity_description])
                entity_name_vdb.insert([entity_name], [entity_name])
                
                already_entities += 1

            # Extract relationships
            relationships = await _handle_json_relationship_extraction(json_data, chunk_key)
            for rel in relationships:
                source_entity = rel["source_entity"]
                target_entity = rel["target_entity"]
                relationship_description = rel["relationship_description"]
                relationship_strength = rel["relationship_strength"]
                source_id = rel["source_id"]
                
                # Add edge to knowledge graph
                knowledge_graph_inst.add_edge(
                    source_entity,
                    target_entity,
                    description=relationship_description,
                    strength=relationship_strength,
                    source_id=source_id,
                )
                
                # Add to relationships vector database
                relationships_vdb.insert([f"{source_entity} -> {target_entity}"], [relationship_description])
                
                already_relations += 1

            already_processed += 1
            print(f"Processed {already_processed} chunks, {already_entities} entities, {already_relations} relations")

        except Exception as e:
            print(f"Error processing chunk {chunk_key}: {e}")
            print(f"Response was: {final_result[:200]}...")

    # Process all chunks
    for chunk_key_dp in ordered_chunks:
        await _process_single_content(chunk_key_dp)

    if already_entities == 0:
        print("WARNING: Didn't extract any entities, maybe your LLM is not working")

    return knowledge_graph_inst
