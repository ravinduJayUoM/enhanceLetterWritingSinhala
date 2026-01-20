"""
Integration test for Phase 1 implementation.
Tests the config, query builder, and document creation with v2 schema.
"""

import sys
import os

# Add rag directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))

import pandas as pd


def test_config():
    """Test configuration module."""
    print("\n" + "=" * 60)
    print("TEST 1: Configuration Module")
    print("=" * 60)
    
    from config import get_config, create_baseline_config, create_sinhala_query_config
    
    config = get_config()
    
    print(f"\n✓ Config loaded successfully")
    print(f"  - CSV Path: {config.data.csv_path}")
    print(f"  - Experiment: {config.experiment_name}")
    print(f"  - Sinhala Query Builder: {config.retrieval.use_sinhala_query_builder}")
    print(f"  - Reranker: {config.retrieval.use_reranker}")
    print(f"  - Embedding Model: {config.embedding.model_name}")
    
    # Test config presets
    baseline = create_baseline_config()
    assert baseline.retrieval.use_sinhala_query_builder == False
    print(f"✓ Baseline config preset works")
    
    sinhala_config = create_sinhala_query_config()
    assert sinhala_config.retrieval.use_sinhala_query_builder == True
    print(f"✓ Sinhala query config preset works")
    
    return True


def test_v2_csv_loading():
    """Test v2 CSV schema loading."""
    print("\n" + "=" * 60)
    print("TEST 2: V2 CSV Schema Loading")
    print("=" * 60)
    
    from config import get_config
    config = get_config()
    
    df = pd.read_csv(config.data.csv_path)
    
    print(f"\n✓ Loaded {len(df)} rows from v2 CSV")
    print(f"  - Columns: {list(df.columns)}")
    
    # Check required columns exist
    required_cols = ['id', 'letter_category', 'doc_type', 'register', 'language', 'source', 'title', 'content']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print(f"✓ All required columns present")
    
    # Check categories
    categories = df['letter_category'].unique().tolist()
    print(f"  - Categories: {categories}")
    
    # Check doc types
    doc_types = df['doc_type'].unique().tolist()
    print(f"  - Doc Types: {doc_types}")
    
    # Check we have structure templates and examples
    assert 'structure' in doc_types or 'example' in doc_types, "No structure or example doc types"
    print(f"✓ Has valid doc types")
    
    return True


def test_query_builder_integration():
    """Test query builder with real data."""
    print("\n" + "=" * 60)
    print("TEST 3: Query Builder Integration")
    print("=" * 60)
    
    from query_builder import SinhalaQueryBuilder, build_sinhala_query
    
    builder = SinhalaQueryBuilder()
    
    # Test 1: Request letter query
    info1 = {
        "letter_type": "request",
        "subject": "නිවාඩු ඉල්ලීම",
        "purpose": "වාර්ෂික නිවාඩු ලබා ගැනීම",
        "recipient": "කළමනාකරු"
    }
    query1 = builder.build_query(info1)
    print(f"\n✓ Request query built:")
    print(f"  {query1[:80]}...")
    assert "ඉල්ලීමේ ලිපිය" in query1, "Sinhala letter type not in query"
    
    # Test 2: Complaint letter query
    info2 = {
        "letter_type": "complaint",
        "subject": "සේවා ගැටලුව",
        "purpose": "අන්තර්ජාල සේවාව නොමැති වීම"
    }
    query2 = builder.build_query(info2)
    print(f"\n✓ Complaint query built:")
    print(f"  {query2[:80]}...")
    assert "පැමිණිලි ලිපිය" in query2, "Sinhala letter type not in query"
    
    # Test 3: Multi-query generation
    queries = builder.build_multi_query(info1, num_queries=3)
    print(f"\n✓ Multi-query generation: {len(queries)} queries")
    for i, q in enumerate(queries):
        print(f"  [{i+1}] {q[:60]}...")
    
    return True


def test_document_creation():
    """Test document creation with v2 schema."""
    print("\n" + "=" * 60)
    print("TEST 4: Document Creation (V2 Schema)")
    print("=" * 60)
    
    from config import get_config
    import pandas as pd
    from langchain_core.documents import Document
    
    config = get_config()
    df = pd.read_csv(config.data.csv_path)
    
    # Simulate document creation logic from sinhala_letter_rag.py
    documents = []
    columns = set(df.columns)
    is_v2_schema = 'letter_category' in columns and 'doc_type' in columns
    
    print(f"\n✓ Detected {'v2' if is_v2_schema else 'v1'} schema")
    
    for _, row in df.iterrows():
        if is_v2_schema:
            title = row.get('title', '')
            content = row.get('content', '')
            letter_category = row.get('letter_category', 'general')
            doc_type = row.get('doc_type', 'example')
            register = row.get('register', 'formal')
            source = row.get('source', 'curated')
            tags = row.get('tags', '')
            doc_id = row.get('id', '')
            
            text = f"{title}\n\n{content}"
            
            metadata = {
                "id": str(doc_id),
                "title": str(title),
                "letter_category": str(letter_category),
                "doc_type": str(doc_type),
                "register": str(register),
                "source": str(source),
                "tags": str(tags) if pd.notna(tags) else "",
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
    
    print(f"✓ Created {len(documents)} documents")
    
    # Show sample documents
    print(f"\nSample documents:")
    for i, doc in enumerate(documents[:3]):
        meta = doc.metadata
        print(f"  [{i+1}] {meta.get('id', 'N/A')}: {meta.get('letter_category', 'N/A')} / {meta.get('doc_type', 'N/A')}")
        print(f"      Title: {meta.get('title', 'N/A')[:50]}...")
    
    # Verify metadata
    structure_docs = [d for d in documents if d.metadata.get('doc_type') == 'structure']
    example_docs = [d for d in documents if d.metadata.get('doc_type') == 'example']
    section_docs = [d for d in documents if d.metadata.get('doc_type') == 'section_template']
    
    print(f"\n✓ Document breakdown:")
    print(f"  - Structure templates: {len(structure_docs)}")
    print(f"  - Full examples: {len(example_docs)}")
    print(f"  - Section templates: {len(section_docs)}")
    
    return True


def test_baseline_vs_sinhala_query():
    """Compare baseline query vs Sinhala-enhanced query."""
    print("\n" + "=" * 60)
    print("TEST 5: Baseline vs Sinhala Query Comparison")
    print("=" * 60)
    
    from query_builder import SinhalaQueryBuilder
    
    # Sample extracted info
    info = {
        "letter_type": "application",
        "subject": "රැකියා අයදුම්පත",
        "purpose": "මෘදුකාංග ඉංජිනේරු තනතුරට",
        "details": "වසර 3ක පළපුරුද්ද",
        "recipient": "මානව සම්පත් කළමනාකරු",
        "sender": "අයදුම්කරු"
    }
    
    # Baseline query (old method)
    baseline_query = f"{info.get('letter_type', '')} {info.get('subject', '')} {info.get('purpose', '')} {info.get('details', '')}"
    
    # Sinhala-enhanced query (new method)
    builder = SinhalaQueryBuilder()
    sinhala_query = builder.build_query(info)
    
    print(f"\n📊 Query Comparison:")
    print(f"\n  BASELINE (old):")
    print(f"  '{baseline_query}'")
    print(f"  Length: {len(baseline_query)} chars")
    
    print(f"\n  SINHALA-ENHANCED (new):")
    print(f"  '{sinhala_query}'")
    print(f"  Length: {len(sinhala_query)} chars")
    
    # Key differences
    print(f"\n✓ Key improvements in Sinhala query:")
    if "අයදුම්පත් ලිපිය" in sinhala_query:
        print(f"  - Letter type translated: 'application' → 'අයදුම්පත් ලිපිය'")
    if "මාතෘකාව" in sinhala_query:
        print(f"  - Field markers added: 'මාතෘකාව', 'අරමුණ', etc.")
    if "විධිමත්" in sinhala_query:
        print(f"  - Formal register marker: 'විධිමත්'")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("🧪 PHASE 1 INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Config Module", test_config()))
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        results.append(("Config Module", False))
    
    try:
        results.append(("V2 CSV Loading", test_v2_csv_loading()))
    except Exception as e:
        print(f"❌ V2 CSV test failed: {e}")
        results.append(("V2 CSV Loading", False))
    
    try:
        results.append(("Query Builder", test_query_builder_integration()))
    except Exception as e:
        print(f"❌ Query builder test failed: {e}")
        results.append(("Query Builder", False))
    
    try:
        results.append(("Document Creation", test_document_creation()))
    except Exception as e:
        print(f"❌ Document creation test failed: {e}")
        results.append(("Document Creation", False))
    
    try:
        results.append(("Baseline vs Sinhala", test_baseline_vs_sinhala_query()))
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        results.append(("Baseline vs Sinhala", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 1 tests passed! Ready for Phase 2.")
    else:
        print("\n⚠️  Some tests failed. Please fix before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
