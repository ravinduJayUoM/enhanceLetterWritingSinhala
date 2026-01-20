"""
Unit tests for the Sinhala Query Builder module.
Tests the query building and letter type mapping functionality.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_builder import (
    SinhalaQueryBuilder,
    build_sinhala_query,
    get_letter_type_sinhala,
    LETTER_TYPE_MAPPING
)


def test_letter_type_mapping():
    """Test English to Sinhala letter type mapping."""
    builder = SinhalaQueryBuilder()
    
    # Test known mappings
    assert builder.map_letter_type("request") == "ඉල්ලීමේ ලිපිය"
    assert builder.map_letter_type("complaint") == "පැමිණිලි ලිපිය"
    assert builder.map_letter_type("application") == "අයදුම්පත් ලිපිය"
    assert builder.map_letter_type("apology") == "ක්ෂමා ලිපිය"
    assert builder.map_letter_type("invitation") == "ආරාධනා ලිපිය"
    assert builder.map_letter_type("general") == "සාමාන්‍ය ලිපිය"
    
    # Test case insensitivity
    assert builder.map_letter_type("REQUEST") == "ඉල්ලීමේ ලිපිය"
    assert builder.map_letter_type("Request") == "ඉල්ලීමේ ලිපිය"
    
    # Test unknown type returns default
    assert builder.map_letter_type("unknown_type") == "සාමාන්‍ය ලිපිය"
    assert builder.map_letter_type("") == "සාමාන්‍ය ලිපිය"
    
    # Test Sinhala input passes through
    assert builder.map_letter_type("ඉල්ලීමේ ලිපිය") == "ඉල්ලීමේ ලිපිය"
    
    print("✓ Letter type mapping tests passed")


def test_query_building():
    """Test query construction from extracted info."""
    builder = SinhalaQueryBuilder()
    
    # Test with full info
    info = {
        "letter_type": "request",
        "subject": "නිවාඩු ඉල්ලීම",
        "purpose": "වාර්ෂික නිවාඩු",
        "details": "පවුලේ කටයුත්තක් සඳහා",
        "recipient": "කළමනාකරු",
        "sender": "සේවක නම"
    }
    
    query = builder.build_query(info)
    
    # Check query contains Sinhala letter type
    assert "ඉල්ලීමේ ලිපිය" in query
    # Check query contains subject
    assert "නිවාඩු ඉල්ලීම" in query
    # Check query contains purpose
    assert "වාර්ෂික නිවාඩු" in query
    
    print(f"✓ Full info query: {query[:100]}...")
    
    # Test with partial info
    partial_info = {
        "letter_type": "complaint",
        "subject": "සේවා ගැටලුව"
    }
    
    partial_query = builder.build_query(partial_info)
    assert "පැමිණිලි ලිපිය" in partial_query
    assert "සේවා ගැටලුව" in partial_query
    
    print(f"✓ Partial info query: {partial_query[:100]}...")
    
    # Test with empty info
    empty_query = builder.build_query({})
    assert len(empty_query) > 0  # Should have at least formal marker
    
    print(f"✓ Empty info query: {empty_query}")


def test_multi_query_building():
    """Test multiple query generation."""
    builder = SinhalaQueryBuilder()
    
    info = {
        "letter_type": "application",
        "subject": "රැකියා අයදුම්පත",
        "purpose": "මෘදුකාංග ඉංජිනේරු තනතුරට"
    }
    
    queries = builder.build_multi_query(info, num_queries=3)
    
    assert len(queries) >= 1
    assert len(queries) <= 3
    
    # Check that queries are unique
    assert len(queries) == len(set(queries))
    
    print(f"✓ Multi-query test passed: {len(queries)} queries generated")
    for i, q in enumerate(queries):
        print(f"  Query {i+1}: {q[:80]}...")


def test_convenience_functions():
    """Test standalone convenience functions."""
    # Test build_sinhala_query
    query = build_sinhala_query(
        letter_type="request",
        subject="නිවාඩු",
        purpose="සෞඛ්‍ය හේතු"
    )
    assert "ඉල්ලීමේ ලිපිය" in query
    
    # Test get_letter_type_sinhala
    sinhala_type = get_letter_type_sinhala("complaint")
    assert sinhala_type == "පැමිණිලි ලිපිය"
    
    print("✓ Convenience functions tests passed")


def test_query_components():
    """Test QueryComponents dataclass."""
    builder = SinhalaQueryBuilder()
    
    info = {
        "letter_type": "invitation",
        "subject": "උත්සව ආරාධනාව",
        "recipient": "සභාපති",
        "sender": "ලේකම්"
    }
    
    components = builder.extract_components(info)
    
    assert components.letter_type == "invitation"
    assert components.letter_type_si == "ආරාධනා ලිපිය"
    assert components.subject == "උත්සව ආරාධනාව"
    assert components.recipient == "සභාපති"
    assert components.sender == "ලේකම්"
    
    # Test to_dict
    comp_dict = components.to_dict()
    assert isinstance(comp_dict, dict)
    assert comp_dict["letter_type_si"] == "ආරාධනා ලිපිය"
    
    print("✓ QueryComponents tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Sinhala Query Builder Tests")
    print("=" * 60 + "\n")
    
    test_letter_type_mapping()
    test_query_building()
    test_multi_query_building()
    test_convenience_functions()
    test_query_components()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
