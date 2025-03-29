"""
Tests for the configuration module.
"""
import os
import pytest
from unittest.mock import patch

from wise_nutrition.utils.config import Config, ConfigurationError

class TestConfig:
    """Test the Config class."""
    
    def test_required_env_variables(self):
        """Test that required environment variables raise an error if missing."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': '',
            'WEAVIATE_URL': 'test-url',
            'WEAVIATE_API_KEY': 'test-key',
            'WEAVIATE_COLLECTION_NAME': 'test-collection'
        }):
            with pytest.raises(ConfigurationError) as exc_info:
                Config()
            assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_optional_env_variables_default_values(self):
        """Test that optional environment variables have default values."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': 'test-key',
            'GOOGLE_API_KEY': 'test-key',
            'WEAVIATE_URL': 'test-url',
            'WEAVIATE_API_KEY': 'test-key',
            'WEAVIATE_COLLECTION_NAME': 'test-collection',
            'OPENAI_MODEL_DEFAULT': 'gpt-3.5-turbo',
            'ANTHROPIC_MODEL_DEFAULT': 'claude-3.5-sonnet-20240620',
            'GOOGLE_MODEL_DEFAULT': 'gemini-2.0-flash-lite',
            'OPENAI_MODEL_HIGH_PERFORMANCE': 'gpt-4o',
            'ANTHROPIC_MODEL_HIGH_PERFORMANCE': 'claude-3-5-sonnet-20241022',
            'GOOGLE_MODEL_HIGH_PERFORMANCE': 'gemini-2.0-flash'
        }):
            config = Config()
            assert config.google_api_key == "test-key"
            assert config.openai_model_default == "gpt-3.5-turbo"
            assert config.anthropic_model_default == "claude-3.5-sonnet-20240620"
    
    def test_get_method_attribute_lookup(self):
        """Test the get method attribute lookup."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': 'test-key',
            'WEAVIATE_URL': 'test-url',
            'WEAVIATE_API_KEY': 'test-key',
            'WEAVIATE_COLLECTION_NAME': 'test-collection'
        }):
            config = Config()
            assert config.get("openai_api_key") == "test-key"
    
    def test_get_method_environment_lookup(self):
        """Test the get method environment lookup."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': 'test-key',
            'WEAVIATE_URL': 'test-url',
            'WEAVIATE_API_KEY': 'test-key',
            'WEAVIATE_COLLECTION_NAME': 'test-collection',
            'CUSTOM_CONFIG': 'custom-value'
        }):
            config = Config()
            assert config.get("CUSTOM_CONFIG") == "custom-value"
    
    def test_get_method_default_value(self):
        """Test the get method default value."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': 'test-key',
            'WEAVIATE_URL': 'test-url',
            'WEAVIATE_API_KEY': 'test-key',
            'WEAVIATE_COLLECTION_NAME': 'test-collection'
        }):
            config = Config()
            assert config.get("MISSING_KEY", "default") == "default"
    