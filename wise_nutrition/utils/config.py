"""
Configuration utilities.
"""
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigurationError(Exception):
    """
    Exception raised for missing required configuration value.
    """
    pass


class Config:
    """
    Application configuration manager.
    """
    
    def __init__(self):
        """
        Initialize the configuration manager.
        
        """
        self._openai_api_key = self._get_required_env_var("OPENAI_API_KEY")
        self._weaviate_url = self._get_required_env_var("WEAVIATE_URL")
        self._weaviate_api_key = self._get_required_env_var("WEAVIATE_API_KEY")
        
        self._weaviate_collection_name = os.getenv("WEAVIATE_COLLECTION_NAME", "nutrition_collection")
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self._openai_model_default = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-3.5-turbo")
        self._anthropic_model_default = os.getenv("ANTHROPIC_MODEL_DEFAULT", "claude-3-5-sonnet-20240620")
        self._google_model_default = os.getenv("GOOGLE_MODEL_DEFAULT", "gemini-2.0-flash-lite")
        self._openai_model_high_performance = os.getenv("OPENAI_MODEL_HIGH_PERFORMANCE", "gpt-4o")
        self._anthropic_model_high_performance = os.getenv("ANTHROPIC_MODEL_HIGH_PERFORMANCE")  
        self._google_model_high_performance = os.getenv("GOOGLE_MODEL_HIGH_PERFORMANCE", "gemini-2.0-flash")

    
    def _get_required_env_var(self, var_name: str) -> str:
        """
        Get a required environment variable.
        """
        value = os.getenv(var_name)
        if value is None or value.strip() == "":
            raise ConfigurationError(f"Missing required environment variable: {var_name}")
        return value
    
    @property
    def openai_api_key(self) -> str:
        """
        Get the OpenAI API key.
        
        Returns:
            OpenAI API key
        """
        return self._openai_api_key
    
    @property
    def anthropic_api_key(self) -> str:
        """
        Get the Anthropic API key.
        
        Returns:
            Anthropic API key
        """
        return self._anthropic_api_key
    
    @property
    def google_api_key(self) -> Optional[str]:
        """
        Get the Google API key.
        
        Returns:
            Google API key or None
        """
        return self._google_api_key
    
    @property
    def openai_model_default(self) -> str:
        """
        Get the default OpenAI model.
        
        Returns:
            Default OpenAI model
        """
        return self._openai_model_default
    
    @property
    def anthropic_model_default(self) -> str:
        """
        Get the default Anthropic model.
        
        Returns:
            Default Anthropic model
        """
        return self._anthropic_model_default
    
    @property
    def google_model_default(self) -> str:
        """
        Get the default Google model.
        
        Returns:
            Default Google model
        """
        return self._google_model_default
    
    @property
    def openai_model_high_performance(self) -> str:
        """
        Get the high-performance OpenAI model.
        
        Returns:
            High-performance OpenAI model
        """
        return self._openai_model_high_performance
    
    @property
    def anthropic_model_high_performance(self) -> str:
        """
        Get the high-performance Anthropic model.
        
        Returns:
            High-performance Anthropic model
        """
        return self._anthropic_model_high_performance
    
    @property
    def google_model_high_performance(self) -> str:
        """
        Get the high-performance Google model.
        """
        return self._google_model_high_performance
    
    @property
    def weaviate_url(self) -> str:
        """
        Get the Weaviate URL.
        
        Returns:
            Weaviate URL
        """
        return self._weaviate_url
    
    @property
    def weaviate_api_key(self) -> str:
        """
        Get the Weaviate API key.
        
        Returns:
            Weaviate API key
        """
        return self._weaviate_api_key
    
    @property
    def weaviate_collection_name(self) -> str:
        """
        Get the Weaviate collection name.
        
        Returns:
            Weaviate collection name
        """
        return self._weaviate_collection_name
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if the key is not found
            
        Returns:
            Configuration value
        """
        # First check if we have the attribute directly
        if hasattr(self, f"_{key}"):
            return getattr(self, f"_{key}")
        
        # Then check environment variables
        env_value = os.getenv(key.upper(), None)
        if env_value is not None:
            # Cache the value for future lookups
            self._config_cache[key] = env_value
            return env_value
            
        # Finally check the cache or return default
        return self._config_cache.get(key, default)
        