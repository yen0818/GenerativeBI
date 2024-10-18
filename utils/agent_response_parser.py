from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing import Literal
import json

class ResponseSchema(BaseModel):
    """Final response to the question being asked"""
    answer_type: Literal["code", "general"]
    answer: str = Field(
        description="The final answer: either a code snippet or a general answer to the question"
    )

# Define a custom parser
class CustomResponseParser:
    def parse(self, text: str) -> ResponseSchema:
        try:
            print(f"Text: {text}")

            if "Final Answer:" in text:
                final_answer = text.split("Final Answer:")[-1].strip()
            else:
                final_answer = text.strip()

            # Ensure the final answer is a JSON string
            try:
                # For Pydantic v2
                response = ResponseSchema.model_validate_json(final_answer)
            except AttributeError:
                # For Pydantic v1
                response = ResponseSchema.parse_raw(final_answer)

            return response
        except (ValidationError, json.JSONDecodeError) as e:
            raise OutputParserException(f"Failed to parse response: {e}")
        except Exception as e:
            raise OutputParserException(f"Unexpected error: {e}")