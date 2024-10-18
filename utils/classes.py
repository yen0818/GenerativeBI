import ast 
import re
import json
import logging
from typing import Tuple, Optional

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def extract_code(response: str) -> Optional[str]:
    """
    Extracts code snippets from the response using markdown code block patterns.
    
    Args:
        response (str): The response string containing code blocks.
        
    Returns:
        Optional[str]: The extracted code if found, else None.
    """
     
    patterns = [
        (r"```python(.*?)```", "python"),
        (r"```py(.*?)```", "py"),
        (r"```json(.*?)```", "json")
    ]

    for pattern, lang in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            if lang == "json":
                json_content = matches[0].strip()
                # Step 2: Parse the JSON content to find the value associated with the "answer" key
                answer_pattern = r'"answer"\s*:\s*"(.*?)"'
                answer_match = re.search(answer_pattern, json_content, re.DOTALL)
                if answer_match:
                    # Step 3: Print the extracted code
                    extracted_code = answer_match.group(1).strip()
                else:
                    extracted_code = json_content
            else:
                extracted_code = matches[0].strip()
            return extracted_code
    
    return response


def extract_dict(response: str) -> Optional[dict]:
    """
    Extracts a dictionary from the response string.
    
    Args:
        response (str): The response string containing a dictionary.
        
    Returns:
        Optional[dict]: The extracted dictionary if found and parsed successfully, else None.
    """
    # Locate the first and last braces
    first_brace = response.find("{")
    last_brace = response.rfind("}")

    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        extracted_string = response[first_brace:last_brace + 1]
        extracted_string = replace_escape_sequences(extracted_string)
        try:
            # Attempt to parse the extracted string as a dictionary
            response_dict = ast.literal_eval(extracted_string)
            if isinstance(response_dict, dict):
                return response_dict
        except (ValueError, SyntaxError) as e:
            return None
            # logger.error(f"Error parsing response dictionary: {e}")
    return None

def replace_escape_sequences(input_string):
    """
    Replaces specific escape sequences in the input string with their actual characters.
    
    Args:
        input_string (str): The string containing escape sequences.
        
    Returns:
        str: The string with escape sequences replaced.
    """

    # Define the pattern to match escape sequences and their replacements
    replacements = {
        r'\\+n': '\n',  # Replace zero or more backslashes followed by 'n' with actual newline
        r'\\+s': ' ',   # Replace zero or more backslashes followed by 's' with actual space
        r"\\+'": "'",   # Replace zero or more backslashes followed by single quote with actual single quote
    }
    
    # Replace each pattern with its corresponding replacement
    for pattern, replacement in replacements.items():
        input_string = re.sub(pattern, replacement, input_string)
    
    return input_string

def post_process(code: str) -> str:
    """
    Cleans up the extracted code by removing unnecessary whitespace and ensuring proper formatting.

    Args:
    code (str): The extracted code string.

    Returns:
    str: The cleaned and formatted code string.
    """
    # Remove unnecessary spaces around semicolons
    code = re.sub(r'\s*;\s*', ';', code)
    # Strip leading and trailing whitespace
    return code.strip()

def format_response(response: str) -> Tuple[str, str]:
    """
    Formats the agent's response by extracting the response type and the corresponding content.
    
    Args:
        response (str): The raw response from the agent.
        
    Returns:
        Tuple[str, str]: A tuple containing the response type and the formatted response content.
    """
    response = replace_escape_sequences(response)
    response_dict = extract_dict(response)

    if response_dict:
        res_type = response_dict.get("answer_type", "general").lower()
        res = response_dict.get("answer", "No answer provided.")
        
        if res_type == "code":
            # Directly return the code without further extraction
            res = post_process(res)
            return res_type, res
        
        if res_type not in ["code", "general"]:
            res_type = "general"
            return res_type, res
        
        # For non-code types, return the general answer
        return res_type, res
    
    # If no dictionary is found, treat the entire response as a general answer
    res_type = "general"
    res = response.strip()
    return res_type, res

def run_request(agent_executor, user_query: str) -> Tuple[str, str]:
    """
    Runs the agent executor with the given user query and processes the response.
    
    Args:
        agent_executor: The agent executor instance.
        user_query (str): The user's query.
        
    Returns:
        Tuple[str, str]: A tuple containing the response type and the formatted response content.
    """
    try:
        response = agent_executor.invoke({"input": user_query})
        # logger.info(f"RESPONSE: {response}")
        llm_response = response.get("output", "")
        res_type, res = format_response(llm_response)
        return res_type, res
    except Exception as e:
        # logger.error(f"Error running request: {e}")
        return "error", f"An error occurred: {e}"