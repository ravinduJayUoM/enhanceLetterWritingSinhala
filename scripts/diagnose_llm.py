"""
Diagnostic script to check LLM configuration and test connectivity.
Run this script to diagnose why /generate-letter might be hanging.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_azure_config():
    """Check Azure OpenAI configuration."""
    print("\n=== Azure OpenAI Configuration ===")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if endpoint:
        print(f"✓ AZURE_OPENAI_ENDPOINT: {endpoint}")
    else:
        print("✗ AZURE_OPENAI_ENDPOINT: NOT SET")
    
    if api_key:
        print(f"✓ AZURE_OPENAI_API_KEY: {'*' * 20} (hidden)")
    else:
        print("✗ AZURE_OPENAI_API_KEY: NOT SET")
    
    if deployment:
        print(f"✓ AZURE_OPENAI_DEPLOYMENT_NAME: {deployment}")
    else:
        print("✗ AZURE_OPENAI_DEPLOYMENT_NAME: NOT SET (will use default: gpt-4-deployment)")
    
    return bool(endpoint and api_key)

def check_config_file():
    """Check the config.py file settings."""
    print("\n=== Configuration File Settings ===")
    
    # Add rag directory to path
    rag_path = os.path.join(os.path.dirname(__file__), 'rag')
    sys.path.insert(0, rag_path)
    
    try:
        from config import get_config, LLMProvider
        config = get_config()
        
        print(f"Current LLM Provider: {config.llm.provider.value}")
        
        if config.llm.provider == LLMProvider.OLLAMA:
            print(f"⚠ WARNING: Using Ollama (local) - URL: {config.llm.ollama_base_url}")
            print(f"  Model: {config.llm.ollama_model}")
            print(f"  This will FAIL in production if Ollama is not running!")
        elif config.llm.provider == LLMProvider.AZURE_OPENAI:
            print(f"✓ Using Azure OpenAI (production recommended)")
        elif config.llm.provider == LLMProvider.OPENAI:
            print(f"Using OpenAI API")
        
        return config
    except Exception as e:
        print(f"✗ Error loading config: {str(e)}")
        return None

def test_llm_connectivity():
    """Test LLM connectivity."""
    print("\n=== Testing LLM Connectivity ===")
    
    rag_path = os.path.join(os.path.dirname(__file__), 'rag')
    sys.path.insert(0, rag_path)
    
    try:
        from sinhala_letter_rag import get_llm
        
        print("Attempting to initialize LLM...")
        llm = get_llm(temperature=0.3)
        print(f"✓ LLM initialized: {type(llm).__name__}")
        
        # Try a simple test
        print("\nTesting with a simple prompt...")
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template("Say 'Hello' in one word.")
        chain = prompt | llm | StrOutputParser()
        
        import asyncio
        from functools import partial
        
        async def test_invoke():
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, partial(chain.invoke, {})),
                timeout=30.0
            )
        
        result = asyncio.run(test_invoke())
        print(f"✓ LLM Response: {result}")
        print("\n✓ LLM is working correctly!")
        return True
        
    except asyncio.TimeoutError:
        print("\n✗ TIMEOUT: LLM did not respond within 30 seconds")
        print("  This is likely the cause of /generate-letter hanging!")
        return False
    except Exception as e:
        print(f"\n✗ Error testing LLM: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks."""
    print("=" * 60)
    print("Sinhala Letter RAG - LLM Configuration Diagnostics")
    print("=" * 60)
    
    # Check Azure configuration
    azure_configured = check_azure_config()
    
    # Check config file
    config = check_config_file()
    
    if not config:
        print("\n✗ Failed to load configuration. Please check config.py")
        return
    
    # Provide recommendations
    print("\n=== Recommendations ===")
    
    from config import LLMProvider
    if config.llm.provider == LLMProvider.OLLAMA and not azure_configured:
        print("⚠ ISSUE DETECTED:")
        print("  - LLM provider is set to Ollama (local)")
        print("  - Ollama is likely not running on the server")
        print("  - This will cause /generate-letter to hang!")
        print("\nSOLUTION:")
        print("  1. Set Azure OpenAI credentials in environment variables")
        print("  2. Or change provider in config.py to LLMProvider.AZURE_OPENAI")
        print("  3. Restart the server")
    elif config.llm.provider == LLMProvider.AZURE_OPENAI and not azure_configured:
        print("⚠ ISSUE DETECTED:")
        print("  - LLM provider is set to Azure OpenAI")
        print("  - But Azure credentials are not set!")
        print("\nSOLUTION:")
        print("  1. Set AZURE_OPENAI_ENDPOINT in environment")
        print("  2. Set AZURE_OPENAI_API_KEY in environment")
        print("  3. Set AZURE_OPENAI_DEPLOYMENT_NAME (optional)")
        print("  4. Restart the server")
    else:
        print("✓ Configuration looks good. Testing connectivity...")
        test_llm_connectivity()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
