"""
RAG chain implementation.
"""
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field # Import Pydantic

from wise_nutrition.utils.prompts import DEFAULT_PROMPT, NUTRITION_BASE_PROMPT

from langchain_openai import ChatOpenAI
from langchain_core.runnables import ( # Updated imports
    Runnable,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable,
    RunnableLambda,
    RunnableConfig,
    RunnableWithMessageHistory
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from uuid import UUID, uuid4

# Import the memory components
from wise_nutrition.memory import ConversationMemoryManager, ConversationState

# Define Input/Output Schemas using Pydantic
class RAGInput(BaseModel):
    query: str = Field(description="The user's query.")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation history.")

class RAGOutput(BaseModel):
    query: str = Field(description="The original user query.")
    response: str = Field(description="The generated response to the query.")
    sources: List[Dict[str, Any]] = Field(description="List of source documents used.")
    structured_data: Dict[str, Any] = Field(description="Extracted structured nutrition data.")
    session_id: str = Field(description="The session ID used or generated.")

# Use composition pattern instead of inheritance
class NutritionRAGChain(BaseModel):
    """
    RAG chain for nutrition-related queries.
    Orchestrates document retrieval, response generation, and source formatting for nutrition advice.
    Uses composition pattern for cleaner Pydantic compatibility.
    """

    retriever: Runnable
    llm: Runnable
    memory_manager: ConversationMemoryManager
    memory_saver: Optional[BaseCheckpointSaver] = None
    model_name: str = "gpt-3.5-turbo"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        retriever: Runnable,
        llm: Runnable,
        memory_manager: ConversationMemoryManager,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        **kwargs
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: A Runnable document retriever instance.
            llm: A Runnable language model instance.
            memory_manager: A ConversationMemoryManager instance.
            openai_api_key: OpenAI API key (optional, used if initializing default LLM).
            model_name: Model name (optional, used if initializing default LLM).
        """
        # Store the memory saver from the manager for easier access
        memory_saver = memory_manager.get_memory_saver()
        
        # Initialize the model with all parameters
        super().__init__(
            retriever=retriever,
            llm=llm,
            memory_manager=memory_manager,
            memory_saver=memory_saver,
            model_name=model_name,
            **kwargs
        )

    # --- Helper methods ---

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format the retrieved documents into a single string context.
        """
        if not docs:
            return "No relevant information found."
        print(f"Formatting {len(docs)} documents.")
        return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))

    def extract_nutrition_data(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Extracts structured nutrition data from retrieved documents.
        """
        print(f"Extracting nutrition data from {len(docs)} documents (placeholder)...")
        # TODO: Implement actual extraction logic
        extracted_data = {}
        return extracted_data

    def format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Formats the source documents into a structured list for citation.
        """
        print("Formatting sources...")
        sources = []
        for i, doc in enumerate(docs):
            source_info = {
                "id": i + 1,
                "content_preview": doc.page_content[:150] + "..." if hasattr(doc, 'page_content') else "N/A"
            }
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                source_info.update(doc.metadata)
            else:
                source_info["metadata"] = "Not available or invalid format"
            sources.append(source_info)
        return sources

    # --- Chain Building ---

    def build_runnable(self) -> Runnable:
        """
        Build the main RAG chain using LCEL, integrated with message history.
        """

        # Define the core RAG steps
        def rag_core_logic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            query = input_dict["query"]
            # History is available via input_dict["history"] if needed for context

            # Simplified analysis/retrieval for core logic
            analyzed_query = query # Use original query directly here

            # Retrieve documents
            try:
                retrieved_docs = self.retriever.invoke(analyzed_query)
            except Exception as e:
                print(f"Error during retrieval in core logic: {e}")
                retrieved_docs = []

            # Format context
            context = self._format_docs(retrieved_docs)

            # Generate response using LLM
            prompt_with_values = NUTRITION_BASE_PROMPT.format_prompt(
                context=context,
                question=query
            )
            try:
                response = self.llm.invoke(prompt_with_values)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                 print(f"Error during LLM invocation: {e}")
                 response_text = "Error generating response."

            # Format sources and structured data
            sources = self.format_sources(retrieved_docs)
            structured_data = self.extract_nutrition_data(retrieved_docs)

            # Return the necessary outputs for the final step
            return {
                "response": response_text,
                "sources": sources,
                "structured_data": structured_data,
                "query": query # Pass query along
            }

        # Wrap the core logic in a RunnableLambda
        _rag_chain_core = RunnableLambda(rag_core_logic)

        # --- Integrate Memory using RunnableWithMessageHistory ---
        chain_with_memory = RunnableWithMessageHistory(
            runnable=_rag_chain_core,
            get_session_history=self._get_session_history_from_manager,
            input_messages_key="query",
            history_messages_key="history",
            output_messages_key="response",
            history_factory_config={
                "session_id": {
                    "id": "session_id",
                    "name": "Session ID",
                    "description": "Unique identifier for the session.",
                    "default": "",
                    "is_shared": True,
                }
            }
        )

        # Final step to format the output into RAGOutput Pydantic model
        def format_final_output(result_dict: Dict[str, Any], config: RunnableConfig) -> RAGOutput:
            session_id = config["configurable"]["session_id"]
            # Ensure all keys exist, provide defaults if necessary
            return RAGOutput(
                query=result_dict.get("query", "<Query not passed>" if not result_dict else "<Query missing>"),
                response=result_dict.get("response", "<Error: No response generated>"),
                sources=result_dict.get("sources", []),
                structured_data=result_dict.get("structured_data", {}),
                session_id=session_id
            )

        # Chain the history wrapper with the final formatting step
        full_chain = chain_with_memory | RunnableLambda(format_final_output)

        return full_chain

    # Wrapper method to fetch history compatible with RWMH
    def _get_session_history_from_manager(self, session_id: str) -> BaseChatMessageHistory:
        """Loads chat history for a given session_id."""
        # For now, return a dummy in-memory history
        # In a production implementation, this would use the memory_saver
        print(f"Warning: Using dummy in-memory ChatMessageHistory for session {session_id}.")
        return ChatMessageHistory()

    # --- Main Interface ---

    def invoke(self, input: RAGInput, config: Optional[RunnableConfig] = None) -> RAGOutput:
        """
        Invoke the RAG chain with history support.
        Accepts RAGInput Pydantic model.
        Returns RAGOutput Pydantic model.
        """
        runnable = self.build_runnable()

        # Ensure config includes session_id for memory
        session_id = input.session_id or self.memory_manager.generate_thread_id()
        final_config = config or {}
        if "configurable" not in final_config:
            final_config["configurable"] = {}
        final_config["configurable"]["session_id"] = session_id

        # Input to RunnableWithMessageHistory expects a dict with the input_messages_key
        history_input = {"query": input.query}

        # Invoke the full chain
        try:
            result = runnable.invoke(history_input, config=final_config)
            
            # Result should already be formatted as RAGOutput by the final lambda
            if isinstance(result, RAGOutput):
                return result
            else:
                # This case indicates an issue in the chain structure or output parsing
                print(f"Error: Unexpected result type from chain: {type(result)}. Expected RAGOutput.")
                return RAGOutput(
                    query=input.query,
                    response="Error: Chain returned unexpected output format.",
                    sources=[],
                    structured_data={},
                    session_id=session_id
                )
        except Exception as e:
            print(f"Error invoking RAG chain: {e}")
            return RAGOutput(
                query=input.query,
                response=f"Error processing your request: {str(e)}",
                sources=[],
                structured_data={},
                session_id=session_id
            )
    
    def as_runnable(self) -> Runnable:
        """Convert this chain to a Runnable object."""
        return RunnableLambda(self.invoke)

# Example usage (for testing, not part of the class):
# if __name__ == "__main__":
#     from langchain_community.retrievers import BM25Retriever
#     from langchain_openai import ChatOpenAI
#
#     # Dummy data and components for testing
#     docs = [Document(page_content="Doc 1 content"), Document(page_content="Doc 2 content")]
#     dummy_retriever = BM25Retriever.from_documents(docs)
#     dummy_llm = ChatOpenAI(model="gpt-3.5-turbo") # Replace with a dummy if needed
#
#     rag_chain_instance = NutritionRAGChain(retriever=dummy_retriever.as_runnable(), llm=dummy_llm)
#     input_data = RAGInput(query="Test query")
#     output = rag_chain_instance.invoke(input_data)
#     print(output) 