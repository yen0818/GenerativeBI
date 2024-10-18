import re
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
from langchain_core.prompts import PromptTemplate

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

TEMPLATE = """
    Use this tool to generate follow-up questions based on the user's chat message if you don't have enough information to answer the question.
    For example, if the user asks "What is the trend of the table?", you can generate a follow-up question like "What specific column would you like to know the trend of?"

    ====== Your Turn =====
    Query: {query}

    """

class FollowUpQuestionInput(BaseModel):
    query: str = Field(description="Chat message from the user")

class FollowUpQuestionTool(BaseTool):
    """Use an LLM to generate follow-up questions from text."""

    name: str = "FollowUpQuestionTool"
    description: str = "Use this tool to generate follow-up questions based on the user's chat message."
    args_schema: Type[BaseModel] = FollowUpQuestionInput
    template: str = TEMPLATE
    llm: BaseLanguageModel
    return_direct: bool = False

    def __init__(self, llm: BaseLanguageModel, **kwargs: Any) -> None:
        super().__init__(llm=llm, **kwargs)
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        prompt = PromptTemplate(template=self.template, input_variables=["query"])
        chain = LLMChain(
                    llm=self.llm,
                    prompt=prompt
                )
        out = chain.predict(query=query, callbacks=run_manager.get_child() if run_manager else None)
        return out

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        prompt = PromptTemplate(template=self.template, input_variables=["query"])
        chain = LLMChain(
                    llm=self.llm,
                    prompt=prompt
                )
        out = await chain.apredict(query=query, callbacks=run_manager.get_child() if run_manager else None)
        return out