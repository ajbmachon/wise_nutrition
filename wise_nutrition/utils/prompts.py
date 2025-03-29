"""
Prompts for the nutrition advisor.
"""

from langchain.prompts import ChatPromptTemplate


NUTRITION_EXPERT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a nutrition expert.
    You are given a question and a list of sources.
    Your task is to answer the question based on the sources.
    """
)




