"""
Prompts for the nutrition advisor.
"""
import os
from langchain.prompts import ChatPromptTemplate
from langsmith import Client
from wise_nutrition.utils.config import Config
from langchain import hub

# client = Client(api_key=Config.langsmith_api_key)
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),

    )

DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a nutrition expert.
    You are given a question and a list of sources.
    Your task is to answer the question based on the sources.
    """
)

# NUTRITION_EXPERT_CHAT_PROMPT = hub.pull("ajbmachon/nutrition_advisor_weston")
# NUTRITION_EXPERT_CHAT_PROMPT = client.pull_prompt("ajbmachon/nutrition_advisor_weston", include_model=True)
prompt = client.pull_prompt("providerprompt/diabetes_risk_assessment", include_model=True)
NUTRITION_EXPERT_CHAT_PROMPT = prompt

