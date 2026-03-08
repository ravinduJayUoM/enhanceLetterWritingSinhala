#!/bin/bash
# Production configuration script for Oracle Cloud deployment
# This script sets the necessary environment variables for Azure OpenAI

echo "=== Sinhala Letter RAG - Production Configuration ==="
echo ""
echo "This script will help you configure the production environment."
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "Found existing .env file. Backing up to .env.backup..."
    cp .env .env.backup
fi

# Create/update .env file
cat > .env << 'EOF'
# Azure OpenAI Configuration (Required for Production)
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here

# Optional: Uncomment to use a different LLM provider
# For Ollama (local): Set LLM_PROVIDER=ollama
# For standard OpenAI: Set LLM_PROVIDER=openai
# LLM_PROVIDER=azure_openai
EOF

echo ""
echo "✓ Created .env template file"
echo ""
echo "IMPORTANT: You need to update the .env file with your actual Azure OpenAI credentials:"
echo "  1. AZURE_OPENAI_ENDPOINT - Your Azure OpenAI endpoint URL"
echo "  2. AZURE_OPENAI_API_KEY - Your Azure OpenAI API key"
echo "  3. AZURE_OPENAI_DEPLOYMENT_NAME - Your deployment name (e.g., gpt-4-deployment)"
echo ""
echo "After updating .env, the application will automatically use Azure OpenAI."
echo ""
echo "To apply changes:"
echo "  1. Edit .env with your credentials"
echo "  2. Restart the server: sudo systemctl restart sinhala-rag"
echo ""
