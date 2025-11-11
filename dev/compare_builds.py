#!/usr/bin/env python3
"""
compare_builds.py - Build Comparison Tool for Invest_AI V3

USAGE:
    python3 dev/compare_builds.py BUILD1 BUILD2
    python3 dev/compare_builds.py V3.0.1+42 V3.0.1+45
    python3 dev/compare_builds.py --list  # List available builds

FEATURES:
    - Compare test results (validator, acceptance, scenario)
    - Diff artifact counts and file lists
    - Side-by-side log comparison (pass/fail status)
    - Detect regressions between builds

OUTPUT:
    Text diff to stdout with colored output (if terminal supports it)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    
    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output"""
        cls.RESET = cls.BOLD = cls.RED = cls.GREEN = ""
        cls.YELLOW = cls.BLUE = cls.CYAN = ""


def load_build_summary(build: str) -> Optional[dict]:
    """Load summary.json for a build."""
    summary_path = Path("dev/builds") / build / "summary.json"
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"{Colors.RED}Error loading {summary_path}: {e}{Colors.RESET}")
        return None


def list_builds():
    """List all available builds."""
    builds_dir = Path("dev/builds")
    if not builds_dir.exists():
        print(f"{Colors.YELLOW}No builds directory found{Colors.RESET}")
        return
    
    builds = sorted([d.name for d in builds_dir.iterdir() if d.is_dir()])
    if not builds:
        print(f"{Colors.YELLOW}No builds found{Colors.RESET}")
        return
    
    print(f"{Colors.BOLD}Available builds:{Colors.RESET}")
    for build in builds:
        summary = load_build_summary(build)
        if summary:
            status = summary.get("overall_status", "UNKNOWN")
            status_color = Colors.GREEN if status == "PASS" else Colors.RED
            timestamp = summary.get("timestamp", "unknown")
            print(f"  {status_color}●{Colors.RESET} {build} ({status}, {timestamp})")
        else:
            print(f"  {Colors.YELLOW}●{Colors.RESET} {build} (no summary)")


def compare_test_results(s1: dict, s2: dict, name: str) -> str:
    """Compare test results between two builds."""
    tests1 = s1.get("tests", {})
    tests2 = s2.get("tests", {})
    
    lines = []
    for test_name in ["validator_pass", "acceptance_pass", "scenario_pass"]:
        r1 = tests1.get(test_name, False)
        r2 = tests2.get(test_name, False)
        
        if r1 == r2:
            status = f"{Colors.GREEN}✓{Colors.RESET}" if r1 else f"{Colors.RED}✗{Colors.RESET}"
            lines.append(f"  {test_name:20s} {status} → {status}")
        else:
            s1_sym = f"{Colors.GREEN}✓{Colors.RESET}" if r1 else f"{Colors.RED}✗{Colors.RESET}"
            s2_sym = f"{Colors.GREEN}✓{Colors.RESET}" if r2 else f"{Colors.RED}✗{Colors.RESET}"
            
            if r2 and not r1:
                lines.append(f"  {test_name:20s} {s1_sym} → {s2_sym} {Colors.GREEN}(FIXED){Colors.RESET}")
            else:
                lines.append(f"  {test_name:20s} {s1_sym} → {s2_sym} {Colors.RED}(REGRESSION){Colors.RESET}")
    
    return "\n".join(lines)


def compare_artifacts(build1: str, build2: str) -> str:
    """Compare artifact directories."""
    art1_dir = Path("dev/builds") / build1 / "artifacts"
    art2_dir = Path("dev/builds") / build2 / "artifacts"
    
    files1 = set(f.name for f in art1_dir.iterdir()) if art1_dir.exists() else set()
    files2 = set(f.name for f in art2_dir.iterdir()) if art2_dir.exists() else set()
    
    only_in_1 = files1 - files2
    only_in_2 = files2 - files1
    common = files1 & files2
    
    lines = [
        f"  Total files: {len(files1)} → {len(files2)}",
        f"  Common: {len(common)}",
    ]
    
    if only_in_1:
        lines.append(f"  {Colors.RED}Only in {build1}: {', '.join(sorted(only_in_1))}{Colors.RESET}")
    
    if only_in_2:
        lines.append(f"  {Colors.GREEN}Only in {build2}: {', '.join(sorted(only_in_2))}{Colors.RESET}")
    
    return "\n".join(lines)


def compare_builds(build1: str, build2: str):
    """Compare two builds and print side-by-side diff."""
    s1 = load_build_summary(build1)
    s2 = load_build_summary(build2)
    
    if not s1:
        print(f"{Colors.RED}Build not found: {build1}{Colors.RESET}")
        return 1
    
    if not s2:
        print(f"{Colors.RED}Build not found: {build2}{Colors.RESET}")
        return 1
    
    # Header
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Build Comparison{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    print(f"{Colors.CYAN}Build 1:{Colors.RESET} {build1}")
    print(f"  Timestamp: {s1.get('timestamp', 'unknown')}")
    print(f"  Status: {s1.get('overall_status', 'UNKNOWN')}")
    print()
    
    print(f"{Colors.CYAN}Build 2:{Colors.RESET} {build2}")
    print(f"  Timestamp: {s2.get('timestamp', 'unknown')}")
    print(f"  Status: {s2.get('overall_status', 'UNKNOWN')}")
    print()
    
    # Test Results Comparison
    print(f"{Colors.BOLD}Test Results:{Colors.RESET}")
    print(compare_test_results(s1, s2, "tests"))
    print()
    
    # Artifacts Comparison
    print(f"{Colors.BOLD}Artifacts:{Colors.RESET}")
    print(compare_artifacts(build1, build2))
    print()
    
    # Overall Assessment
    status1 = s1.get("overall_status")
    status2 = s2.get("overall_status")
    
    print(f"{Colors.BOLD}Overall:{Colors.RESET}")
    if status1 == status2:
        print(f"  Status unchanged: {status1}")
    elif status2 == "PASS" and status1 != "PASS":
        print(f"  {Colors.GREEN}IMPROVEMENT: {status1} → {status2}{Colors.RESET}")
    else:
        print(f"  {Colors.RED}REGRESSION: {status1} → {status2}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Invest_AI V3 builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s V3.0.1+42 V3.0.1+45       Compare two specific builds
  %(prog)s --list                    List all available builds
        """
    )
    
    parser.add_argument(
        "builds",
        nargs="*",
        help="Two build names to compare (e.g., V3.0.1+42 V3.0.1+45)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available builds"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    # Handle --list
    if args.list:
        list_builds()
        return 0
    
    # Validate arguments
    if len(args.builds) != 2:
        parser.print_help()
        print(f"\n{Colors.RED}Error: Provide exactly two build names or use --list{Colors.RESET}")
        return 1
    
    build1, build2 = args.builds
    return compare_builds(build1, build2)


if __name__ == "__main__":
    sys.exit(main())
