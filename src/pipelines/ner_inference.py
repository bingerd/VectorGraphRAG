from vertexai.generative_models import GenerativeModel

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

def extract_entities_and_relations(model: GenerativeModel,text: str):
    """
    Ultra-intelligent entity and relation extraction using an LLM.
    Returns:
      entities: [{"name": str, "type": str}]
      relations: [{"subject": str, "predicate": str, "object": str, "metadata": {...}}]
    """

    prompt = f"""
    You are an expert information extraction model.
    Analyze the text below and extract:
      1. All named entities (with clear, semantic types: PERSON, ORG, LOCATION, EVENT, PRODUCT, etc.)
      2. All semantic relations between entities in subject-predicate-object format.

    Return the result as strict JSON with this structure:
    {{
      "entities": [{{"name": "...", "type": "..."}}],
      "relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]
    }}

    Text:
    \"\"\"{text}\"\"\"
    """
    response = model.generate_content(
        contents=prompt, 
        generation_config={"response_mime_type": "application/json", "temperature": 0}
        )
    
    try:
        import json
        data = json.loads(response.text)
        entities = data.get("entities", [])
        relations = data.get("relations", [])
    except Exception:
        entities, relations = [], []
    # logger.info(entities, relations)
    return entities, relations