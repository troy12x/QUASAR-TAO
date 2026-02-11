# The MIT License (MIT)
# Copyright 2026 SILX INC
#
# Tests for context_builder module

import os
import tempfile
from pathlib import Path
from quasar.utils.context_builder import (
    build_full_context,
    build_minimal_context,
    generate_file_tree,
    validate_repo_structure,
    estimate_context_tokens,
)


def test_file_tree_generation():
    """Test file tree generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create test structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "chunk.py").write_text("# chunk.py content")
        (repo_path / "src" / "kernels").mkdir()
        (repo_path / "src" / "kernels" / "attention.cu").write_text("// CUDA code")
        (repo_path / "__init__.py").write_text("# init")
        
        tree = generate_file_tree(repo_path)
        assert "chunk.py" in tree
        assert "attention.cu" in tree
        print("✅ File tree generation test passed")


def test_minimal_context():
    """Test minimal context building."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create test files
        chunk_file = repo_path / "chunk.py"
        chunk_file.write_text("def chunk_quasar(): pass")
        
        gate_file = repo_path / "gate.py"
        gate_file.write_text("def gate(): pass")
        
        context = build_minimal_context(str(repo_path), "chunk.py", ["chunk.py", "gate.py"])
        
        assert "chunk.py" in context
        assert "chunk_quasar" in context
        assert "gate.py" in context
        print("✅ Minimal context test passed")


def test_full_context():
    """Test full context building."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create test structure
        (repo_path / "src").mkdir()
        chunk_file = repo_path / "src" / "chunk.py"
        chunk_file.write_text("def chunk_quasar(): pass")
        
        gate_file = repo_path / "src" / "gate.py"
        gate_file.write_text("def gate(): pass")
        
        (repo_path / "src" / "kernels").mkdir()
        cuda_file = repo_path / "src" / "kernels" / "attention.cu"
        cuda_file.write_text("// CUDA kernel code")
        
        context = build_full_context(str(repo_path), "chunk.py", include_tree=True)
        
        assert "QUASAR FULL REPOSITORY CONTEXT" in context
        assert "chunk.py" in context
        assert "chunk_quasar" in context
        assert "STRICT INSTRUCTIONS" in context
        
        tokens = estimate_context_tokens(context)
        assert tokens > 0
        print(f"✅ Full context test passed (estimated {tokens} tokens)")


def test_repo_validation():
    """Test repository validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Empty repo should have warnings
        is_valid, warnings = validate_repo_structure(str(repo_path))
        assert not is_valid
        assert len(warnings) > 0
        
        # Add required files
        (repo_path / "chunk.py").write_text("# chunk")
        (repo_path / "__init__.py").write_text("# init")
        (repo_path / "kernel.cu").write_text("// cuda")
        
        is_valid, warnings = validate_repo_structure(str(repo_path))
        assert is_valid or len(warnings) == 0
        print("✅ Repository validation test passed")


if __name__ == "__main__":
    print("Running context_builder tests...")
    test_file_tree_generation()
    test_minimal_context()
    test_full_context()
    test_repo_validation()
    print("\n✅ All tests passed!")
