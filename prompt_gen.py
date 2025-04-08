"""Prompt generation for the Loop Runner.

Handles creating prompts for LLMs based on linter results.
"""

from typing import Sequence
from models import CompletedLinterRun


def generate_prompt(script: str, results: Sequence[CompletedLinterRun]) -> str:
    """Generate a prompt for the LLM based on linter results.

    Args:
        script: The script of linters on which the results are based
        results: List of completed linter runs to include in prompt

    Returns:
        Formatted prompt string for LLM
    """
    prompt = ["Here are some source files we will be working with"]

    # Add source files
    for result in results:
        prompt.extend([
            f"<<<< {result.path} <<<<",
            result.content,
            ">>>>>>>>"
        ])

    # Add linter output
    with open(script, 'r') as f:
        script_content = f.read()
    prompt.extend([
        "We will process these with the following script",
        f"<<<< {script} <<<<\n{script_content}\n>>>>>>>>",
        "This script is applied with the following command line",
        f"{script} {' '.join(r.path for r in results)}",
        "yielding the following, with exit status 1",
        "<<<< stdout <<<<",
        "\n".join(r.stdout for r in results if r.stdout),
        ">>>>>>>>",
        "<<<< stderr <<<<",
        "\n".join(r.stderr for r in results if r.stderr),
        ">>>>>>>>",
        "",
        "Given all the preceding, here are instructions about what to do next",
        "<<<< instructions <<<<",
        "Please suggest fixes for any issues identified by the linters.",
        "For each file that needs changes, provide the complete new content",
        "in a markdown code block with the filename as a level-3 header.",
        "Every function and class definition should be preceded by two blank lines.",
        "Every function, class, and module should get a docstring.",
        "Explain your changes in the text before each code block.",
        "If no changes are needed for a file, do not include it.",
        "Here is an example of how to specify complete new content for a file",
        "where 'path/to/foo.py' is a placeholder for the actual path",
        "to the file you are fixing:",
        "",
        "### path/to/foo.py",
        "",
        "```python",
        "def foo():",
        "    return 'bar'",
        "```",
        "",
        ">>>>>>>>",
    ])

    return "\n".join(prompt)
