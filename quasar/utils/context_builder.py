
# Context Builder for QUASAR-SUBNET
# Builds rich repository context for LLM code generation
#
# This module provides full repository visibility to the model during generation,
# ensuring it understands the complete codebase contract and exports required functions.

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUASAR - CONTEXT BUILDER                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE                                                                     ║
║  ───────                                                                     ║
║  Provides full repository context to LLM during code generation, ensuring    ║
║  the model understands the complete codebase contract and exports required   ║
║  functions (e.g., chunk_quasar).                                            ║
║                                                                              ║
║  FEATURES                                                                    ║
║  ────────                                                                    ║
║  - Full file tree generation                                                 ║
║  - Smart file filtering (critical patterns + keyword matching)               ║
║  - Token-aware context building (prevents overflow)                          ║
║  - Support for BYOC (Bring-Your-Own-Code) mode                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple
from collections import Counter


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONFIGURATION                                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Critical patterns that must always be included
CRITICAL_PATTERNS = [
    "chunk",
    "quasar",
    "attention",
    "kernel",
    "gate",
    "fused",
    "forward",
    "recurrent"
]

# Critical keywords that indicate important files
CRITICAL_KEYWORDS = [
    "chunk_quasar",
    "quasar_attention",
    "flash_attention",
    "linear_attention",
    "__init__"
]

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".py": True,
    ".cu": True,  # CUDA source files
    ".cuh": True,  # CUDA header files
    ".h": True,  # C/C++ header files
    ".hpp": True,  # C++ header files
}

# Files/directories to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    "*.egg-info",
    ".ipynb_checkpoints",
    "*.pyc",
    ".mypy_cache",
]

# Maximum context size (in characters, approximate)
# DeepSeek-V3.2 supports ~128K tokens, but we'll be conservative
MAX_CONTEXT_SIZE = 200000  # ~50K tokens


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FILE TREE GENERATION                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def generate_file_tree(repo_path: Path, max_depth: int = 4) -> str:
    """
    Generate a text representation of the repository file tree.
    
    Args:
        repo_path: Root path of the repository
        max_depth: Maximum depth to traverse (prevents huge trees)
    
    Returns:
        String representation of the file tree
    """
    lines = []
    
    def should_exclude(path: Path) -> bool:
        """Check if path should be excluded."""
        path_str = str(path)
        return any(pattern in path_str for pattern in EXCLUDE_PATTERNS)
    
    def tree_recursive(path: Path, prefix: str = "", depth: int = 0):
        """Recursively build tree representation."""
        if depth > max_depth:
            return
        
        if should_exclude(path):
            return
        
        # Get all items, sorted
        try:
            items = sorted([p for p in path.iterdir() if not should_exclude(p)], 
                          key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            lines.append(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir():
                extension = "    " if is_last else "│   "
                tree_recursive(item, prefix + extension, depth + 1)
    
    tree_recursive(repo_path)
    return "\n".join(lines)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FILE FILTERING                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def is_critical_file(file_path: Path, target_file: str = "chunk.py") -> bool:
    """
    Check if a file is critical and must be included.
    
    Args:
        file_path: Path to the file
        target_file: Target file being edited (e.g., "chunk.py")
    
    Returns:
        True if file is critical
    """
    name_lower = file_path.name.lower()
    path_str = str(file_path).lower()
    
    # Always include target file
    if file_path.name == target_file:
        return True
    
    # Check critical patterns in filename
    if any(pattern in name_lower for pattern in CRITICAL_PATTERNS):
        return True
    
    # Check for critical keywords in path
    if any(keyword in path_str for keyword in CRITICAL_KEYWORDS):
        return True
    
    return False


def file_contains_keywords(file_path: Path, keywords: List[str]) -> bool:
    """
    Check if file contains any of the specified keywords.
    
    Args:
        file_path: Path to the file
        keywords: List of keywords to search for
    
    Returns:
        True if file contains any keyword
    """
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
        return any(keyword.lower() in content for keyword in keywords)
    except Exception:
        return False


def score_file_relevance(file_path: Path, target_file: str = "chunk.py") -> float:
    """
    Score file relevance (higher = more relevant).
    
    Args:
        file_path: Path to the file
        target_file: Target file being edited
    
    Returns:
        Relevance score (0.0 to 1.0)
    """
    score = 0.0
    name_lower = file_path.name.lower()
    path_str = str(file_path).lower()
    
    # Critical files get highest score
    if is_critical_file(file_path, target_file):
        score += 10.0
    
    # Check for imports/exports of target file
    if target_file.replace(".py", "") in name_lower:
        score += 5.0
    
    # Check for critical keywords in content
    if file_contains_keywords(file_path, CRITICAL_KEYWORDS):
        score += 3.0
    
    # Prefer files in src/, kernels/, or similar directories
    if any(dir_name in path_str for dir_name in ["src", "kernels", "quasar", "attention"]):
        score += 2.0
    
    # Prefer Python files (more relevant for API)
    if file_path.suffix == ".py":
        score += 1.0
    
    return score


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONTEXT BUILDING                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def collect_relevant_files(
    repo_path: Path,
    target_file: str = "chunk.py",
    max_files: int = 50,
    max_size: int = MAX_CONTEXT_SIZE
) -> List[Tuple[Path, float]]:
    """
    Collect relevant files from repository with relevance scoring.
    
    Args:
        repo_path: Root path of the repository
        target_file: Target file being edited
        max_files: Maximum number of files to include
        max_size: Maximum total size (characters)
    
    Returns:
        List of (file_path, score) tuples, sorted by relevance
    """
    files_with_scores = []
    total_size = 0
    
    # First pass: collect all candidate files
    for ext, include in INCLUDE_EXTENSIONS.items():
        if not include:
            continue
        
        for file_path in repo_path.rglob(f"*{ext}"):
            # Skip excluded files
            if any(pattern in str(file_path) for pattern in EXCLUDE_PATTERNS):
                continue
            
            # Skip if too large (individual file limit)
            try:
                file_size = file_path.stat().st_size
                if file_size > 50000:  # Skip files > 50KB
                    continue
            except Exception:
                continue
            
            # Score file
            score = score_file_relevance(file_path, target_file)
            files_with_scores.append((file_path, score))
    
    # Sort by relevance (highest first)
    files_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Second pass: select files up to size limit
    selected_files = []
    current_size = 0
    
    # Always include critical files first
    critical_files = [(fp, score) for fp, score in files_with_scores if score >= 10.0]
    for file_path, score in critical_files[:max_files]:
        try:
            file_size = len(file_path.read_text(encoding='utf-8', errors='ignore'))
            if current_size + file_size <= max_size:
                selected_files.append((file_path, score))
                current_size += file_size
        except Exception:
            continue
    
    # Then add other relevant files
    other_files = [(fp, score) for fp, score in files_with_scores if score < 10.0]
    for file_path, score in other_files:
        if len(selected_files) >= max_files:
            break
        
        try:
            file_size = len(file_path.read_text(encoding='utf-8', errors='ignore'))
            if current_size + file_size <= max_size:
                selected_files.append((file_path, score))
                current_size += file_size
        except Exception:
            continue
    
    return selected_files


def format_file_content(file_path: Path, repo_path: Path) -> str:
    """
    Format file content with header.
    
    Args:
        file_path: Path to the file
        repo_path: Root path of the repository
    
    Returns:
        Formatted file content string
    """
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        relative_path = file_path.relative_to(repo_path)
        
        # Format with header
        header = f"\n# File: {relative_path}\n"
        return header + content
    except Exception as e:
        return f"\n# File: {file_path.relative_to(repo_path)}\n# Error reading file: {e}\n"


def build_full_context(
    repo_path: str,
    target_file: str = "chunk.py",
    include_tree: bool = True,
    max_files: int = 50,
    max_size: int = MAX_CONTEXT_SIZE,
    byoc_mode: bool = False,
    byoc_file_path: Optional[str] = None
) -> str:
    """
    Build full repository context for LLM generation.
    
    Args:
        repo_path: Path to the repository root
        target_file: Target file being edited (e.g., "chunk.py")
        include_tree: Whether to include file tree
        max_files: Maximum number of files to include
        max_size: Maximum context size (characters)
        byoc_mode: Bring-Your-Own-Code mode (expert miner provides optimized code)
        byoc_file_path: Path to expert's optimized file (if byoc_mode=True)
    
    Returns:
        Formatted context string
    """
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    context_parts = []
    
    # Header
    context_parts.append("# ════════════════════════════════════════════════════════════════════════")
    context_parts.append("# QUASAR FULL REPOSITORY CONTEXT")
    context_parts.append("# ════════════════════════════════════════════════════════════════════════")
    context_parts.append("")
    
    # File tree
    if include_tree:
        context_parts.append("# File Tree:")
        context_parts.append("```")
        try:
            tree = generate_file_tree(repo_path)
            context_parts.append(tree)
        except Exception as e:
            context_parts.append(f"# Error generating tree: {e}")
        context_parts.append("```")
        context_parts.append("")
    
    # BYOC mode: Include expert's optimized code first
    if byoc_mode and byoc_file_path:
        byoc_path = Path(byoc_file_path)
        if byoc_path.exists():
            context_parts.append("# ════════════════════════════════════════════════════════════════════════")
            context_parts.append("# ⚠️ EXPERT CODE PROVIDED - USE AS PRIMARY REFERENCE")
            context_parts.append("# ════════════════════════════════════════════════════════════════════════")
            context_parts.append("")
            context_parts.append("IMPORTANT: Expert code is shown below. Use it as your main reference.")
            context_parts.append("Adapt structure/imports to match repository, but keep the core implementation.")
            context_parts.append("")
            context_parts.append(format_file_content(byoc_path, repo_path))
            context_parts.append("")
    
    # Relevant source files
    context_parts.append("# ════════════════════════════════════════════════════════════════════════")
    context_parts.append("# RELEVANT SOURCE FILES")
    context_parts.append("# ════════════════════════════════════════════════════════════════════════")
    context_parts.append("")
    
    # Collect and format files
    relevant_files = collect_relevant_files(repo_path, target_file, max_files, max_size)
    
    for file_path, score in relevant_files:
        formatted = format_file_content(file_path, repo_path)
        context_parts.append(formatted)
        context_parts.append("")  # Blank line between files
    
    # Simplified, actionable instructions
    context_parts.append("# ════════════════════════════════════════════════════════════════════════")
    context_parts.append("# KEY REQUIREMENTS")
    context_parts.append("# ════════════════════════════════════════════════════════════════════════")
    context_parts.append("")
    context_parts.append("When generating code, ensure:")
    context_parts.append("")
    context_parts.append("✓ Function signature matches codebase (check __init__.py and imports)")
    context_parts.append("✓ chunk_quasar function is exported correctly")
    context_parts.append("✓ Imports match repository structure (use existing imports only)")
    context_parts.append("✓ Code style matches repository patterns")
    context_parts.append("")
    if byoc_mode and byoc_file_path:
        context_parts.append("⚠️ EXPERT CODE PROVIDED:")
        context_parts.append("   → Use expert code implementation as your reference")
        context_parts.append("   → Adapt structure/imports to match repository, but keep core logic")
        context_parts.append("")
    context_parts.append("")
    
    return "\n".join(context_parts)


def build_minimal_context(
    repo_path: str,
    target_file: str = "chunk.py",
    specific_files: Optional[List[str]] = None
) -> str:
    """
    Build minimal context with only specified files (for testing/debugging).
    
    Args:
        repo_path: Path to the repository root
        target_file: Target file being edited
        specific_files: List of specific file names to include (relative to repo root)
    
    Returns:
        Formatted context string
    """
    repo_path = Path(repo_path)
    context_parts = []
    
    context_parts.append("# QUASAR MINIMAL CONTEXT")
    context_parts.append("")
    
    if specific_files:
        for file_name in specific_files:
            file_path = repo_path / file_name
            if file_path.exists():
                context_parts.append(format_file_content(file_path, repo_path))
                context_parts.append("")
    
    # Always include target file if it exists
    target_path = repo_path / target_file
    if target_path.exists() and target_file not in (specific_files or []):
        context_parts.append(format_file_content(target_path, repo_path))
        context_parts.append("")
    
    return "\n".join(context_parts)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ UTILITY FUNCTIONS                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def estimate_context_tokens(context: str) -> int:
    """
    Estimate token count for context (rough approximation).
    
    Args:
        context: Context string
    
    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for code
    return len(context) // 4


def validate_repo_structure(repo_path: str) -> Tuple[bool, List[str]]:
    """
    Validate repository structure and check for required files.
    
    Args:
        repo_path: Path to the repository root
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    repo_path = Path(repo_path)
    warnings = []
    
    if not repo_path.exists():
        return False, [f"Repository path does not exist: {repo_path}"]
    
    # Check for critical files
    critical_files = ["chunk.py", "__init__.py"]
    for file_name in critical_files:
        file_path = repo_path / file_name
        if not file_path.exists():
            warnings.append(f"Critical file not found: {file_name}")
    
    # Check for CUDA files
    cuda_files = list(repo_path.rglob("*.cu"))
    if not cuda_files:
        warnings.append("No CUDA files (.cu) found in repository")
    
    return len(warnings) == 0, warnings
