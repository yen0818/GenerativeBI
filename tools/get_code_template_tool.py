import re, os
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field
from langchain.chains.llm import LLMChain

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.prompts import PromptTemplate

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_openai import  AzureOpenAIEmbeddings 
from langchain_core.documents import Document

from const import EMBED_MODEL
from const import FAISS_TEMPLATE_FILE_PATH, TEMPLATE_FILE_PATH

import json

from dotenv import load_dotenv
load_dotenv()

# Initialize the embeddings and FAISS vector store as before
embed_model = AzureOpenAIEmbeddings(deployment=EMBED_MODEL, model=EMBED_MODEL)

# Create or load the FAISS vector store
if os.path.exists(FAISS_TEMPLATE_FILE_PATH):
    faiss_vector_store = FAISS.load_local(FAISS_TEMPLATE_FILE_PATH, embed_model, allow_dangerous_deserialization=True)
else:
    with open(TEMPLATE_FILE_PATH, "r") as file:
        data = json.load(file)
    documents = []
    for chart_name, chart_info in data['charts'].items():
        doc_content = f"Chart Name: {chart_name}\nChart Description: {chart_info['description']}\nCode Template: {chart_info['code']}\nInstructions in generating the chart: {chart_info['instructions']}"
        documents.append(Document(page_content=doc_content))
    for table_name, table_info in data['tables'].items():
        doc_content = f"Table Type: {table_name}\nTable Description: {table_info['description']}\nCode Template: {table_info['code']}"
        documents.append(Document(page_content=doc_content))
    faiss_vector_store = FAISS.from_documents(documents, embed_model)
    faiss_vector_store.save_local(FAISS_TEMPLATE_FILE_PATH)

# Get the VectorStoreRetriever from the FAISS vector store
faiss_vector_store_retriever = faiss_vector_store.as_retriever()

def get_code_templates(requirements: Union[str, List[str]]   
) -> str:
    """
    Retrieve the code templates from the FAISS vector store.
    
    Args:
        requirements (Union[str, List[str]]): The requirements of the task.
    
    Returns:
        str: The code templates.
    """
    return faiss_vector_store_retriever.invoke(requirements)

class GetCodeTemplateInput(BaseModel):
    requirements: str = Field(description="The requirements of the task.")

class GetCodeTemplateTool(BaseTool):
    """Use an LLM to generate a plan for a coding task."""

    name: str = "GetCodeTemplateTool"
    description: str = "Use this tool to get code templates for generating a plan for a coding task."
    args_schema: Type[BaseModel] = GetCodeTemplateInput

    def _run(
        self, requirements: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return get_code_templates(requirements)

    async def _arun(
        self,
        requirements: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(requirements=requirements, run_manager=run_manager)