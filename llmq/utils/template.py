"""Template resolution utilities for LLMQ."""

import json
from typing import Any, Dict, List
import logging


logger = logging.getLogger(__name__)


def resolve_template_string(template: str, data: Dict[str, Any]) -> str:
    """Resolve template variables in a string using data."""
    try:
        return template.format(**data)
    except KeyError as e:
        logger.warning(f"Template variable {e} not found in data")
        return template


def resolve_template_messages(
    template_messages: List[Dict[str, Any]], data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Resolve template variables in messages using data."""
    resolved_messages = []

    for message in template_messages:
        if isinstance(message, dict):
            resolved_message = {}
            for key, value in message.items():
                if isinstance(value, str):
                    resolved_message[key] = resolve_template_string(value, data)
                else:
                    resolved_message[key] = value
            resolved_messages.append(resolved_message)
        else:
            resolved_messages.append(message)

    return resolved_messages


def format_json_template(json_obj: Any, data: Dict[str, Any]) -> Any:
    """Recursively format JSON template with data values."""
    if isinstance(json_obj, str):
        return resolve_template_string(json_obj, data)
    elif isinstance(json_obj, dict):
        return {
            key: format_json_template(value, data) for key, value in json_obj.items()
        }
    elif isinstance(json_obj, list):
        return [format_json_template(value, data) for value in json_obj]
    else:
        return json_obj


def validate_required_fields(data: Dict[str, Any], template_str: str) -> List[str]:
    """Extract and validate required template variables from a string."""
    import re

    # Find all {variable} patterns
    variables = re.findall(r"\{([^}]+)\}", template_str)
    missing_vars = []

    for var in variables:
        if var not in data:
            missing_vars.append(var)

    return missing_vars


def create_job_from_data(
    item: Dict[str, Any],
    index: int,
    column_mapping: Dict[str, str],
    job_id_prefix: str = "job",
) -> Dict[str, Any]:
    """Create a job dictionary from data item using column mapping."""
    import uuid

    job_data: Dict[str, Any] = {
        "id": f"{job_id_prefix}-{index:08d}-{uuid.uuid4().hex[:8]}"
    }

    # Apply column mapping
    for job_field, mapping_value in column_mapping.items():
        logger.debug(f"Processing mapping: {job_field} = {mapping_value}")

        if (mapping_value.startswith("{") and mapping_value.endswith("}")) or (
            mapping_value.startswith("[") and mapping_value.endswith("]")
        ):
            # Handle JSON mapping for complex fields like messages
            try:
                json_template = json.loads(mapping_value)
                job_data[job_field] = format_json_template(json_template, item)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in mapping for field '{job_field}': {mapping_value}. Error: {e}"
                )
                continue
        elif "{" in mapping_value and "}" in mapping_value:
            # Handle template string mapping
            missing_vars = validate_required_fields(item, mapping_value)
            if missing_vars:
                logger.warning(
                    f"Template variables {missing_vars} not found in data item for field '{job_field}'"
                )
            job_data[job_field] = resolve_template_string(mapping_value, item)
        elif mapping_value in item:
            # Simple column mapping
            job_data[job_field] = item[mapping_value]
        else:
            logger.warning(
                f"Column '{mapping_value}' not found in data item. Available columns: {list(item.keys())}"
            )

    return job_data


def ensure_job_has_prompt_or_messages(
    job_data: Dict[str, Any], item: Dict[str, Any]
) -> Dict[str, Any]:
    """Ensure job has either prompt or messages field."""
    if "messages" not in job_data and "prompt" not in job_data:
        # Fallback: use text column as prompt if available
        if "text" in item:
            job_data["prompt"] = str(item["text"])
        else:
            raise ValueError(
                f"No messages or prompt could be created from item. Available keys: {list(item.keys())}"
            )

    # Set chat_mode=True if we have messages
    if "messages" in job_data and job_data["messages"] is not None:
        job_data["chat_mode"] = True

    return job_data
