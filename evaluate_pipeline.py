"""
Pipeline Evaluation Script
Tests each component of the RAG pipeline and compares baseline vs enhanced approaches
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))

from rag.sinhala_letter_rag import LetterDatabase, RAGProcessor, UserQuery, get_llm
from rag.config import get_config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Test prompts in Sinhala
TEST_PROMPTS = [
    {
        "name": "Leave Request",
        "prompt": "මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න.",
        "expected_type": "request"
    },
    {
        "name": "Job Application",
        "prompt": "මම ගුණසේකර රාජකීය විද්‍යාලයේ ඉංග්‍රීසි ගුරු තනතුරට අයදුම් කිරීමට කැමතියි. මට බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද ඇත.",
        "expected_type": "application"
    },
    {
        "name": "Complaint Letter",
        "prompt": "ජනවාරි 15 වන දින මා විසින් ඔබගේ ආයතනයෙන් මිළදී ගත් රෙෆ්‍රිජරේටරය නිසි ලෙස ක්‍රියා නොකරයි. කරුණාකර මෙය හදා දෙන්න හෝ ආපසු ගෙවන්න.",
        "expected_type": "complaint"
    }
]

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---\n")

def test_extraction(rag_processor, test_prompt):
    """Test information extraction"""
    print_subsection(f"Testing Extraction: {test_prompt['name']}")
    print(f"Prompt: {test_prompt['prompt']}\n")
    
    extracted = rag_processor.extract_key_info(test_prompt['prompt'])
    
    print("Extracted Information:")
    for key, value in extracted.items():
        if value and value != "N/A":
            print(f"  {key}: {value}")
    
    return extracted

def test_retrieval(rag_processor, extracted_info):
    """Test document retrieval"""
    print_subsection("Testing Retrieval")
    
    docs = rag_processor.retrieve_relevant_content(extracted_info, top_k=3)
    
    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        category = doc.metadata.get('letter_category', 'unknown')
        doc_type = doc.metadata.get('doc_type', 'unknown')
        title = doc.metadata.get('title', 'untitled')
        print(f"  [{i}] {category} - {doc_type}")
        print(f"      Title: {title}")
        print(f"      Content preview: {doc.page_content[:100]}...\n")
    
    return docs

def generate_baseline_letter(llm, test_prompt):
    """Generate letter without RAG (baseline)"""
    print_subsection("Baseline Generation (No RAG)")
    
    baseline_prompt = ChatPromptTemplate.from_template("""You are a Sinhala formal letter writing assistant.

Write a complete formal letter in Sinhala based on this request:
{prompt}

Instructions:
- Write ONLY in Sinhala script
- Use proper formal letter structure
- Include appropriate greetings and closings
- Use formal register

Generate the letter:""")
    
    chain = baseline_prompt | llm | StrOutputParser()
    letter = chain.invoke({"prompt": test_prompt['prompt']})
    
    print("Generated Letter (Baseline):")
    print(letter)
    print(f"\nLength: {len(letter)} characters")
    
    return letter

def generate_enhanced_letter(rag_processor, llm, test_prompt, extracted_info, retrieved_docs):
    """Generate letter with RAG enhancement"""
    print_subsection("Enhanced Generation (With RAG)")
    
    enhanced_prompt = rag_processor.construct_enhanced_prompt(
        test_prompt['prompt'],
        extracted_info,
        retrieved_docs,
        None
    )
    
    print("Enhanced Prompt Preview:")
    print(enhanced_prompt[:500] + "...\n")
    
    letter_prompt = ChatPromptTemplate.from_template("{enhanced_prompt}")
    chain = letter_prompt | llm | StrOutputParser()
    
    letter = chain.invoke({"enhanced_prompt": enhanced_prompt})
    
    print("Generated Letter (Enhanced):")
    print(letter)
    print(f"\nLength: {len(letter)} characters")
    
    return letter, enhanced_prompt

def evaluate_letter_quality(letter, test_name):
    """Basic quality checks for generated letter"""
    print_subsection(f"Quality Evaluation: {test_name}")
    
    checks = {
        "Has Sinhala content": any('\u0d80' <= c <= '\u0dff' for c in letter),
        "Has English translation": "translation:" in letter.lower() or "letter:" in letter.lower(),
        "Reasonable length": 200 <= len(letter) <= 2000,
        "Has formal greeting": any(word in letter for word in ["මහතා", "මහත්මිය", "ගරු"]),
        "Has formal closing": any(word in letter for word in ["ස්තූතියි", "ඔබගේ", "අනුමත"]),
    }
    
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check}")
    
    score = sum(checks.values()) / len(checks) * 100
    print(f"\nOverall Score: {score:.1f}%")
    
    return score

def compare_approaches(baseline_letter, enhanced_letter):
    """Compare baseline vs enhanced approach"""
    print_subsection("Comparison: Baseline vs Enhanced")
    
    metrics = {
        "Length": (len(baseline_letter), len(enhanced_letter)),
        "Sinhala density": (
            sum(1 for c in baseline_letter if '\u0d80' <= c <= '\u0dff'),
            sum(1 for c in enhanced_letter if '\u0d80' <= c <= '\u0dff')
        )
    }
    
    for metric, (baseline_val, enhanced_val) in metrics.items():
        diff = enhanced_val - baseline_val
        pct_change = (diff / baseline_val * 100) if baseline_val > 0 else 0
        print(f"  {metric}:")
        print(f"    Baseline: {baseline_val}")
        print(f"    Enhanced: {enhanced_val}")
        print(f"    Change: {diff:+d} ({pct_change:+.1f}%)")

def run_full_evaluation():
    """Run complete pipeline evaluation"""
    print_section("SINHALA LETTER RAG PIPELINE EVALUATION")
    
    # Initialize components
    print("Initializing components...")
    config = get_config()
    print(f"Using LLM provider: {config.llm.provider}")
    print(f"Model: {config.llm.ollama_model if config.llm.provider.value == 'OLLAMA' else config.llm.openai_model}")
    print(f"Sinhala Query Builder: {config.retrieval.use_sinhala_query_builder}")
    print(f"Reranker: {config.retrieval.use_reranker}")
    
    letter_db = LetterDatabase()
    letter_db.build_knowledge_base()
    rag_processor = RAGProcessor(letter_db)
    llm = get_llm(temperature=0.3)
    
    print(f"Knowledge base size: {letter_db.get_document_count()} documents")
    
    # Test each prompt
    results = []
    for test_prompt in TEST_PROMPTS:
        print_section(f"TEST CASE: {test_prompt['name']}")
        
        # Step 1: Extraction
        extracted_info = test_extraction(rag_processor, test_prompt)
        
        # Step 2: Retrieval
        retrieved_docs = test_retrieval(rag_processor, extracted_info)
        
        # Step 3: Baseline generation
        baseline_letter = generate_baseline_letter(llm, test_prompt)
        baseline_score = evaluate_letter_quality(baseline_letter, "Baseline")
        
        # Step 4: Enhanced generation
        enhanced_letter, enhanced_prompt = generate_enhanced_letter(
            rag_processor, llm, test_prompt, extracted_info, retrieved_docs
        )
        enhanced_score = evaluate_letter_quality(enhanced_letter, "Enhanced")
        
        # Step 5: Comparison
        compare_approaches(baseline_letter, enhanced_letter)
        
        results.append({
            "name": test_prompt['name'],
            "baseline_score": baseline_score,
            "enhanced_score": enhanced_score,
            "improvement": enhanced_score - baseline_score
        })
    
    # Final summary
    print_section("EVALUATION SUMMARY")
    
    print("Results by Test Case:")
    for result in results:
        improvement_str = f"{result['improvement']:+.1f}%"
        print(f"\n  {result['name']}:")
        print(f"    Baseline Score:  {result['baseline_score']:.1f}%")
        print(f"    Enhanced Score:  {result['enhanced_score']:.1f}%")
        print(f"    Improvement:     {improvement_str}")
    
    avg_baseline = sum(r['baseline_score'] for r in results) / len(results)
    avg_enhanced = sum(r['enhanced_score'] for r in results) / len(results)
    avg_improvement = avg_enhanced - avg_baseline
    
    print(f"\nOverall Averages:")
    print(f"  Baseline:    {avg_baseline:.1f}%")
    print(f"  Enhanced:    {avg_enhanced:.1f}%")
    print(f"  Improvement: {avg_improvement:+.1f}%")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if avg_improvement < 5:
        print("""
⚠️  RAG pipeline shows minimal improvement. Consider:

1. MODEL ISSUES:
   - llama3.2:3b is very small (2GB) and struggles with Sinhala
   - Switch to aya:8b (4.8GB) - specifically trained on Sinhala
   - Command: ollama pull aya:8b
   - Update config.py: ollama_model = "aya:8b"

2. DATA ISSUES:
   - Only 12 documents in knowledge base is too small
   - Add 50-100 more diverse letter examples
   - Ensure examples cover all letter categories

3. PROMPT ENGINEERING:
   - Current prompts may not be optimal for small models
   - Try simpler, more direct instructions
   - Reduce the length of example templates

4. RETRIEVAL ISSUES:
   - Check if retrieved documents are actually relevant
   - Enable reranker (set use_reranker=True in config.py)
   - Verify embeddings work well for Sinhala queries
""")
    else:
        print(f"""
✓ RAG pipeline shows {avg_improvement:.1f}% improvement!

Continue with:
1. Test with reranker enabled (Phase 3)
2. Expand knowledge base to 50-100 examples
3. Consider switching to aya:8b for better Sinhala support
4. Run quantitative evaluation for thesis research
""")

if __name__ == "__main__":
    try:
        run_full_evaluation()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
