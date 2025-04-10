"""
Prompts for the nutrition advisor.
"""
import os
from langchain.prompts import ChatPromptTemplate
# from langsmith import Client # Commented out LangSmith for now
from wise_nutrition.utils.config import Config
# from langchain import hub # Commented out LangSmith hub for now

# client = Client(api_key=Config.langsmith_api_key)
# client = Client(
#     api_key=os.getenv("LANGSMITH_API_KEY"),
#     )

DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a nutrition expert.
    You are given a question and a list of sources.
    Your task is to answer the question based on the sources.
    """
)

NUTRITION_BASE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert nutrition advisor specializing in Weston A. Price Foundation principles.
Your goal is to provide accurate, helpful, and evidence-based nutrition advice based *only* on the provided context.

**Instructions:**
1.  Answer the user's question using *only* the information found in the 'Context' section below.
2.  If the context does not contain enough information to answer the question, state that clearly.
3.  Do *not* make up information or provide advice outside the scope of the provided context.
4.  Cite the sources for your answer where appropriate (Source formatting will be handled separately).
5.  Your advice is for informational purposes only and is not medical advice. Include a brief disclaimer at the end of your response reminding the user of this.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""
)

# Placeholder for future specialized prompts
# MEAL_PLANNING_PROMPT = ChatPromptTemplate.from_template("...")
# NUTRIENT_INFO_PROMPT = ChatPromptTemplate.from_template("...")

# Removed LangSmith hub pulls for now
# NUTRITION_EXPERT_CHAT_PROMPT = hub.pull("ajbmachon/nutrition_advisor_weston")
# NUTRITION_EXPERT_CHAT_PROMPT = client.pull_prompt("ajbmachon/nutrition_advisor_weston", include_model=True)
# prompt = client.pull_prompt("providerprompt/diabetes_risk_assessment", include_model=True)
# NUTRITION_EXPERT_CHAT_PROMPT = prompt

