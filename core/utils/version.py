"""Build version stamping utility.

Provides get_build_version() that returns a semantic version string
derived from git describe or commit count since latest tag, with
optional ".dirty" suffix for uncommitted changes.

Safe fallbacks to timestamp when git is unavailable or repo is untagged.
No external dependencies beyond stdlib.
"""

from __future__ import annotations
import subprocess
import os
from datetime import datetime
from typing import Optional


def get_build_diagnostics(repo_path: Optional[str] = None) -> dict:
    """
    Get detailed build diagnostics for debugging.
    
    Returns:
        Dict with commit_hash, git_describe, commits_since_tag, tracked_dirty, untracked_count
    """
    if repo_path is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            current = script_dir
            for _ in range(5):
                if os.path.exists(os.path.join(current, ".git")):
                    repo_path = current
                    break
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
        except Exception:
            pass
    
    result = {
        "commit_hash": None,
        "git_describe": None,
        "commits_since_tag": None,
        "tracked_dirty": False,
        "untracked_count": 0,
    }
    
    try:
        # Short commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if hash_result.returncode == 0:
            result["commit_hash"] = hash_result.stdout.strip()
        
        # Git describe
        describe_result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if describe_result.returncode == 0:
            result["git_describe"] = describe_result.stdout.strip()
        
        # Commits since v3.0.1
        count_result = subprocess.run(
            ["git", "rev-list", "--count", "v3.0.1..HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if count_result.returncode == 0:
            result["commits_since_tag"] = int(count_result.stdout.strip())
        
        # Tracked dirty files
        DEVNULL = open(os.devnull, 'w')
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_path,
                stderr=DEVNULL,
                timeout=2
            )
            result["tracked_dirty"] = len(dirty.strip()) > 0
        finally:
            DEVNULL.close()
        
        # Untracked count
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if untracked_result.returncode == 0:
            result["untracked_count"] = len([l for l in untracked_result.stdout.strip().split("\n") if l])
    except Exception:
        pass
    
    return result


def get_build_version(repo_path: Optional[str] = None) -> str:
    """
    Get build version string in format: "V3.0.1+N" or "V3.0.1+N.dirty"
    
    Strategy:
      1. Try `git describe --tags --always --dirty` for full version
      2. Fall back to `git rev-list --count HEAD` + latest tag for commit count
      3. Fall back to timestamp if git unavailable or repo untagged
    
    Args:
        repo_path: Path to git repository (default: current working directory)
    
    Returns:
        Version string like "V3.0.1+42.dirty" or "dev-20251110-143000"
    """
    if repo_path is None:
        # Try to find repo root from this file's location
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Walk up to find .git directory
            current = script_dir
            for _ in range(5):  # Max 5 levels up
                if os.path.exists(os.path.join(current, ".git")):
                    repo_path = current
                    break
                parent = os.path.dirname(current)
                if parent == current:  # Reached filesystem root
                    break
                current = parent
        except Exception:
            pass
    
    # Try git describe first (most comprehensive)
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            describe = result.stdout.strip()
            # Format: v3.0.1-42-g1a2b3c4-dirty
            # Convert to: V3.0.1+42.dirty
            if describe.startswith("v"):
                describe = "V" + describe[1:]
            # Replace -N-gHASH with +N
            parts = describe.split("-")
            if len(parts) >= 3 and parts[-2].startswith("g"):
                # Has commit distance: v3.0.1-42-g1a2b3c4[-dirty]
                version = parts[0]
                distance = parts[1]
                dirty = ".dirty" if "dirty" in parts else ""
                return f"{version}+{distance}{dirty}"
            elif "dirty" in describe:
                # Just tag with dirty: v3.0.1-dirty
                return describe.replace("-dirty", ".dirty")
            else:
                # Just tag or commit hash
                return describe
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Try commit count + latest tag
    try:
        # Get latest tag
        tag_result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        
        # Get commit count since tag
        count_result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        
        # Check for uncommitted changes (ignore untracked files)
        DEVNULL = open(os.devnull, 'w')
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_path,
                stderr=DEVNULL,
                timeout=2
            )
            is_dirty = len(dirty.strip()) > 0
        finally:
            DEVNULL.close()
        
        if tag_result.returncode == 0 and count_result.returncode == 0:
            tag = tag_result.stdout.strip()
            if tag.startswith("v"):
                tag = "V" + tag[1:]
            count = count_result.stdout.strip()
            dirty = ".dirty" if is_dirty else ""
            return f"{tag}+{count}{dirty}"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Final fallback: timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"dev-{timestamp}"


if __name__ == "__main__":
    # Quick test
    version = get_build_version()
    print(f"Build version: {version}")
