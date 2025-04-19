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
from wise_nutrition.citation_generator import CitationGenerator, Citation

# Define Input/Output Schemas using Pydantic
class RAGInput(BaseModel):
    query: str = Field(description="The user's query.")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation history.")

class RAGOutput(BaseModel):
    query: str = Field(description="The original user query.")
    response: str = Field(description="The generated response to the query.")
    sources: List[Dict[str, Any]] = Field(description="List of source documents used.")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Formatted citations for sources.")
    structured_data: Dict[str, Any] = Field(description="Extracted structured nutrition data.")
    session_id: str = Field(description="The session ID used or generated.")

# Implement Runnable interface for LangServe compatibility
class NutritionRAGChain(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """
    RAG chain for nutrition-related queries.
    Orchestrates document retrieval, response generation, and source formatting for nutrition advice.
    Implements RunnableSerializable for LangServe compatibility.
    """

    retriever: Runnable
    llm: Runnable
    memory_manager: ConversationMemoryManager
    citation_generator: CitationGenerator
    memory_saver: Optional[BaseCheckpointSaver] = None
    model_name: str = "gpt-3.5-turbo"
    
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        retriever: Runnable,
        llm: Runnable,
        memory_manager: ConversationMemoryManager,
        citation_generator: Optional[CitationGenerator] = None,
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
            citation_generator: A CitationGenerator instance (optional).
            openai_api_key: OpenAI API key (optional, used if initializing default LLM).
            model_name: Model name (optional, used if initializing default LLM).
        """
        # Store the memory saver from the manager for easier access
        memory_saver = memory_manager.get_memory_saver()
        
        # Create a default citation generator if none provided
        if citation_generator is None:
            citation_generator = CitationGenerator()
        
        # Initialize the model with all parameters
        super().__init__(
            retriever=retriever,
            llm=llm,
            memory_manager=memory_manager,
            citation_generator=citation_generator,
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

    def generate_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate formatted citations for the source documents.
        
        Args:
            docs: List of source documents
            
        Returns:
            List of citation dictionaries
        """
        print("Generating citations...")
        citations = self.citation_generator.generate_citations(docs)
        
        # Convert Citation objects to dictionaries for output
        citation_dicts = []
        for i, citation in enumerate(citations):
            citation_dict = {
                "id": i + 1,
                "text": citation.text,
                "source_name": citation.source_name,
                "source_url": citation.source_url,
                "date_accessed": citation.date_accessed,
                "preview": citation.original_content
            }
            citation_dicts.append(citation_dict)
            
        return citation_dicts

    # --- Runnable Interface Implementation ---
    
    def invoke(self, input_data: RAGInput, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Invoke the RAG chain with the given input data.
        
        Args:
            input_data: RAGInput model containing the query and optional session_id
            config: Optional configuration for the runnable
            
        Returns:
            Dictionary containing the response and related information
        """
        # Extract query and session_id from input
        query = input_data.query
        session_id = input_data.session_id
        
        if not query.strip():
            return {
                "query": query,
                "response": "Please provide a valid query.",
                "sources": [],
                "citations": [],
                "structured_data": {},
                "session_id": session_id or str(uuid4())
            }
            
        # Get or create conversation state
        conversation = self.memory_manager.get_conversation_state(session_id)
        if session_id is None:
            session_id = conversation.session_id
            
        # Get conversation history
        history = conversation.get_messages()
        
        # Process using core logic
        result = self._rag_core_logic({"query": query, "history": history, "session_id": session_id})
        
        # Update conversation with new messages
        conversation.add_user_message(query)
        conversation.add_ai_message(result["response"])
        
        # Save conversation state
        self.memory_manager.save_conversation_state(conversation)
        
        # Return formatted result
        return {
            "query": query,
            "response": result["response"],
            "sources": result["sources"],
            "citations": result["citations"],
            "structured_data": result["structured_data"],
            "session_id": session_id
        }
    
        # Update conversation with new messages
        conversation.add_user_message(query)
        conversation.add_ai_message(result["response"])
        
        # Save conversation state
        self.memory_manager.save_conversation_state(conversation)
        
        # Return formatted result
        return {
            "query": query,
            "response": result["response"],
            "sources": result["sources"],
            "citations": result["citations"],
            "structured_data": result["structured_data"],
            "session_id": session_id
        }
    
    # --- Core Logic ---

    def _rag_core_logic(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        query = input_dict["query"]
        history = input_dict.get("history", [])
        session_id = input_dict.get("session_id", str(uuid4()))

        # Filter history to prevent context window overflow
        filtered_history = self.memory_manager.filter_messages(history)

        # Retrieve documents
        try:
            retrieved_docs = self.retriever.invoke(query)
        except Exception as e:
            print(f"Error during retrieval in core logic: {e}")
            retrieved_docs = []

        # Format context
        context = self._format_docs(retrieved_docs)

        # Generate response using LLM with filtered history
        prompt_with_values = NUTRITION_BASE_PROMPT.format_prompt(
            context=context,
            question=query
        )
        
        # Combine filtered history with current prompt for LLM
        messages_for_llm = filtered_history + [prompt_with_values.to_messages()[0]]

        # Generate response with the LLM
        llm_response = self.llm.invoke(messages_for_llm)
        response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Extract structured data (placeholder for now)
        structured_data = self.extract_nutrition_data(retrieved_docs)
        
        # Format sources for citation
        sources = self.format_sources(retrieved_docs)
        
        # Generate citations
        citations = self.generate_citations(retrieved_docs)
        
        # Return the complete result
        return {
            "query": query,
            "response": response_text,
            "sources": sources,
            "citations": citations,
            "structured_data": structured_data,
            "session_id": session_id
        }

    def as_runnable(self) -> Runnable:
        """Convert this chain to a Runnable object."""
        return self
#     # Dummy data and components for testing
#     docs = [Document(page_content="Doc 1 content"), Document(page_content="Doc 2 content")]
#     dummy_retriever = BM25Retriever.from_documents(docs)
#     dummy_llm = ChatOpenAI(model="gpt-3.5-turbo") # Replace with a dummy if needed
#
#     rag_chain_instance = NutritionRAGChain(retriever=dummy_retriever.as_runnable(), llm=dummy_llm)
#     input_data = RAGInput(query="Test query")
#     output = rag_chain_instance.invoke(input_data)
#     print(output) 