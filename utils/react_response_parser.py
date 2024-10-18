import re
import logging
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

from const import (
    FINAL_ANSWER_ACTION,
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE
    )

logger = logging.getLogger(__name__)

class ReActFinalAnswerOnlyParser(AgentOutputParser):
    """Parses ReAct-style LLM calls and extracts only the Final Answer."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentFinish:
        includes_answer = FINAL_ANSWER_ACTION in text
        
        if includes_answer:
            # Extract everything after 'Final Answer:'
            final_answer = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": final_answer}, text)
        
        # If Final Answer is not present, raise an exception
        raise OutputParserException(
            f"Could not find '{FINAL_ANSWER_ACTION}' in the response: `{text}`",
            observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
            llm_output=text,
            send_to_llm=True,
        )

    @property
    def _type(self) -> str:
        return "react-final-answer-only"


class ReActSingleInputOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                logger.info(f"Parse Error (Include Final answer and action error): {text}")
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            # Output is without 'Final Answer:', 
            # text contains both observation and Final Answer:, 
            # hence we need the splitting to include only the final answer
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            print("\n====IF LOOP MISIING ACTION======\n")
            logger.info(f"Parse Error (Missing Action): {text}")
            # TODO: Raise exception. If all logged text are final answer, will change to AgentFinish instead
            raise OutputParserException(
                f"Could not parse LLM output here. Might be missing Final Answer or Thought: `{text}`, please try again.",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            print("\n====IF LOOP MISIING ACTION INPUT======\n")
            logger.info(f"Parse Error (Missing Action Input): {text}")
            raise OutputParserException(
                f"Could not parse LLM output. Missing Action Input: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            print("\n====IF LOOP MISIING OTHERS======\n")
            logger.info(f"Parse Error (Missing Others): {text}")
            raise OutputParserException(f"Could not parse LLM output. OTHERS: `{text}`")

    @property
    def _type(self) -> str:
        return "react-single-input"