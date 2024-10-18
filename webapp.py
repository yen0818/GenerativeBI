from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_experimental.tools import PythonAstREPLTool

from utils.react_response_parser import ReActSingleInputOutputParser, ReActFinalAnswerOnlyParser
from utils.ingestion import data_ingestion
from utils.agent_response_parser import CustomResponseParser
from const import GPT_MODEL, PROMPT_TEMPLATE, FORMAT_INSTRUCTIONS, GENERATE_PLAN_TEMPLATE
from tools.follow_up_question_tool import FollowUpQuestionTool
from tools.generate_plan_tool import GeneratePlanTool
from tools.get_code_template_tool import GetCodeTemplateTool

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI language model
llm = AzureChatOpenAI(deployment_name=GPT_MODEL, model_name=GPT_MODEL, temperature=0)

# Initialize custom parser for the response
custom_parser = CustomResponseParser()

# Streamlit app configuration
st.set_page_config(page_title="GenBI", page_icon="📊", layout="wide")

if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

if "k" not in st.session_state:
    st.session_state.k = 5

with st.sidebar:
    try:
        data = st.file_uploader("\U0001F4BB Load a CSV file:", type="csv")
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))

    st.write("\U0001F9F9 Clear message history:")
    clear_history = st.sidebar.button("Clear message history")
    
    st.sidebar.write("\U00002699 Agent Settings:")
    st.session_state.k = st.sidebar.slider("**Memory size**", 1, 10, st.session_state.k)

if not data:
    st.write("\U0001F916 **Upload a CSV file to get started.**")
else:
    file_name = data.name
    df, db, engine, schema = data_ingestion(filename=data)
    df_columns = df.columns.tolist()
    df_description = df.describe()
    df_info = df.info()

    st.write("##### Data Preview")
    st.dataframe(df.head())

    # Prompt template
    prompt_template = PromptTemplate.from_template(
        PROMPT_TEMPLATE, 
        partial_variables={"table_schema": schema, "format_instructions": FORMAT_INSTRUCTIONS, "generate_plan_instructions": GENERATE_PLAN_TEMPLATE}
    )

    if 'memory' not in st.session_state:
    # conversational agent memory
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=st.session_state.k,
            return_messages=True
        )
    
    # Initialize tools for the agent
    tools = [
        FollowUpQuestionTool(llm=llm),
        PythonAstREPLTool(locals={"df": df}),
        # GeneratePlanTool(llm=llm, df=df, df_columns=df_columns)
        GetCodeTemplateTool()
    ]

    # Create ReAct agent with custom output parser
    react_agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=prompt_template, 
        output_parser=ReActSingleInputOutputParser()
    )

    # Initialize agent executor
    agent_executor = AgentExecutor(
        agent=react_agent, 
        tools=tools, 
        verbose=True,
        memory=st.session_state.memory,
        handle_parsing_errors=True,
        max_iteration=1
        )
    
    if "messages" not in st.session_state or clear_history:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                content = message["content"]
                st.write(content)

    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:

        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Execute the agent
                    response_text = agent_executor.invoke({"input": user_query})

                    # Parse the response
                    response = custom_parser.parse(response_text['output'])
                    print(f"Response: {response}")

                    st.session_state.messages.append({"role": "assistant", "content": response.answer})

                    # Handle the response based on its type
                    if response.answer_type == "code":
                        try:
                            exec(response.answer, {"df": df, "st": st})
                            st.write("✅ Code executed successfully.")
                        except Exception as e:
                            st.error(f"An error occurred while executing the code: {e}")
                    elif response.answer_type == "general":
                        st.write(response.answer)
                    else:
                        st.error("Unknown answer type received.")
                except ValueError as e:
                    st.error(f"Parsing error: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            

    