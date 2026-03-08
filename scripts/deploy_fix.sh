#!/bin/bash
# Quick deployment script for the /generate_letter fix

echo "================================================"
echo "Deploying /generate_letter Hanging Fix"
echo "================================================"
echo ""

# Step 1: Confirm we're in the right directory
if [ ! -f "run_server.py" ]; then
    echo "❌ Error: run_server.py not found. Are you in the project root?"
    exit 1
fi

echo "✓ In project directory"
echo ""

# Step 2: Show what changed
echo "📝 Files modified:"
echo "  - rag/sinhala_letter_rag.py (main fix)"
echo "  - rag/config.py (kept Ollama as default)"
echo ""

# Step 3: Check if service exists
if systemctl list-units --type=service --all | grep -q "sinhala-rag.service"; then
    echo "✓ Found sinhala-rag service"
    
    # Step 4: Restart the service
    echo ""
    echo "🔄 Restarting service..."
    sudo systemctl restart sinhala-rag
    
    if [ $? -eq 0 ]; then
        echo "✓ Service restarted successfully"
    else
        echo "❌ Failed to restart service"
        exit 1
    fi
    
    # Step 5: Check status
    echo ""
    echo "📊 Service status:"
    sudo systemctl status sinhala-rag --no-pager | head -20
    
    # Step 6: Show recent logs
    echo ""
    echo "📋 Recent logs (last 20 lines):"
    sudo journalctl -u sinhala-rag -n 20 --no-pager
    
else
    echo "⚠ Service not found. Starting manually..."
    echo ""
    echo "Run: python3 run_server.py"
fi

echo ""
echo "================================================"
echo "✓ Deployment complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Test with: python3 test_generate_letter_fix.py"
echo "  2. Or use curl to test the endpoint"
echo "  3. Monitor logs: sudo journalctl -u sinhala-rag -f"
echo ""
