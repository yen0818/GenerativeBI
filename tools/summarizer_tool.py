import re
from typing import Any, Dict, List, Optional, Type, Union
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain.chains.llm import LLMChain

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

import warnings
import pandas as pd

def check_type(dtype: str, value):
    """Cast value to right type to ensure it is JSON serializable"""
    if "float" in str(dtype):
        return float(value)
    elif "int" in str(dtype):
        return int(value)
    else:
        return value
def get_column_properties(df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
    """Get properties of each column in a pandas DataFrame"""
    properties_list = []
    for column in df.columns:
        dtype = df[column].dtype
        properties = {}
        if dtype in [int, float, complex]:
            properties["dtype"] = "number"
            properties["std"] = check_type(dtype, df[column].std())
            properties["min"] = check_type(dtype, df[column].min())
            properties["max"] = check_type(dtype, df[column].max())

        elif dtype == bool:
            properties["dtype"] = "boolean"
        elif dtype == object:
            # Check if the string column can be cast to a valid datetime
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(df[column], errors='raise')
                    properties["dtype"] = "date"
            except ValueError:
                # Check if the string column has a limited number of values
                if df[column].nunique() / len(df[column]) < 0.5:
                    properties["dtype"] = "category"
                else:
                    properties["dtype"] = "string"
        elif pd.api.types.is_categorical_dtype(df[column]):
            properties["dtype"] = "category"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            properties["dtype"] = "date"
        else:
            properties["dtype"] = str(dtype)

        # add min max if dtype is date
        if properties["dtype"] == "date":
            try:
                properties["min"] = df[column].min()
                properties["max"] = df[column].max()
            except TypeError:
                cast_date_col = pd.to_datetime(df[column], errors='coerce')
                properties["min"] = cast_date_col.min()
                properties["max"] = cast_date_col.max()
        # Add additional properties to the output dictionary
        nunique = df[column].nunique()
        if "samples" not in properties:
            non_null_values = df[column][df[column].notnull()].unique()
            n_samples = min(n_samples, len(non_null_values))
            samples = pd.Series(non_null_values).sample(
                n_samples, random_state=42).tolist()
            properties["samples"] = samples
        properties["num_unique_values"] = nunique
        properties["semantic_type"] = ""
        properties["description"] = ""
        properties_list.append(
            {"column": column, "properties": properties})

    return properties_list

# class SummarizerInput(BaseModel):
#     df: pd.DataFrame = Field(description="The dataframe to summarize")

class SummarizerTool(BaseTool):
    """Use an LLM to generate summarization from a dataset."""

    name: str = "SummarizerTool"
    description: str = """Use this tool to annotate datasets and generate summarization when user asks for a summary of the dataset or to understand the dataset better. 
    The input is a dataframe. The output is a list of dictionaries, each containing the properties of a column in the dataframe. 
    The properties include the data type, number of unique values, and a few samples of the column values.
    Convert the output to a string representation of a list of dictionaries to return the final answer.

    <assistant> Thought: The user asking for a summary of the dataset. I should use the SummarizerTool to generate a summary of the dataset.
    <assistant> Action: SummarizerTool
    <assistant> Action Input: df
    <assistant> This is a summary of the dataset: (replace this with the summary generated)
    """
    # args_schema: Type[BaseModel] = SummarizerInput
    df: pd.DataFrame = Field(exclude=True)

    def _run(
        self,
        *args, **kwargs
        # df: pd.DataFrame,
    ) -> str:
        """Use the tool."""
        return get_column_properties(self.df)

    async def _arun(
        self,
        *args, **kwargs
        # df: pd.DataFrame,
    ) -> str:
        """Use the tool asynchronously."""
        return get_column_properties(self.df)