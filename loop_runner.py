#!/usr/bin/env python
"""Code Quality Auto-Fix Loop Runner

This script automates the process of fixing code quality issues by:
1. Running code quality checks (e.g. ruff, pylint, flake8, pytest)
2. Asking an LLM (running on Ollama) to fix any issues found
3. Applying the suggested fixes
4. Repeating until all checks pass

The script can use either:
- Groq's API for fast, cloud-based inference
- Docker to run a local Ollama LLM server

Code checks are run directly on the host machine.

Usage:
    python loop_runner.py [options] <filenames>

The script will continue running until either:
- All checks pass successfully
- Maximum iterations reached
- LLM fails to provide valid fixes

The LLM is expected to respond with complete file contents in
Markdown format, which are then written back to disk:

### path/to/file.py
```python
# Complete corrected file content
```

The LLM runs in a Docker container. Code checks and patching are
done directly on the host machine.
"""

import os
import sys
import re
import asyncio
import argparse
import logging
from datetime import datetime
import requests
import docker
from docker.errors import APIError
from llm_wrangler import LLMWrangler, Platform
from prompt_gen import generate_prompt
from git_wrangler import GitWrangler
from models import LinterRun, CompletedLinterRun, ProposedFix

MINUTE: float = 60.0

logger = logging.getLogger("loop_runner")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s\n%(message)s'
))
logger.addHandler(handler)

BRIEF_INTRO = """\
This script automates the process of fixing code quality issues by:
1. Running code quality checks (e.g. ruff, pylint, flake8, pytest)
2. Asking an LLM (running on Ollama) to fix any issues found
3. Applying the suggested fixes
4. Repeating until all checks pass
The LLM runs in a Docker container. Code checks and patching are
done directly on the host machine.
"""

PROMPT_TEMPLATE = """\
# Please fix the following issues in some Python source files

## Source files

{% for file in files %}
### {{ file }}

```python
{{ lookup[file].content }}
```

#### Linter stdout
{{ lookup[file].stdout }}

#### Linter stderr
{{ lookup[file].stderr }}
{% endfor %}

## Instructions

1. DO NOT DISCARD ANY DOCSTRINGS OR COMMENTS.
2. If docstrings or comments are misleading, propose a fix for them.
3. If any docstrings are missing, add them.
4. Fill in any missing type hints or annotations.

## Your response should be in Markdown format, looking like this:

### path/to/file.py

Provide an explanation of the issue as you see it and how you
propose to fix it. Use a triple-backtick-python block to show the
new version of the file. Show the ENTIRE CONTENT of the file, not
just a diff. Remember, "path/to/file.py" is a placeholder. Replace
it with the actual path to the file.
"""


class CustomHelpFormatter(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter
):
    """Custom formatter that shows defaults and
    preserves description formatting

    Args:
        argparse.RawDescriptionHelpFormatter:
            Formatter that preserves description formatting
        argparse.ArgumentDefaultsHelpFormatter:
            Formatter that shows defaults
    """


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        argparse.Namespace: Parsed command line arguments containing sources, debug level,
                          model name, iterations, linter-script, and llm-url
    """
    parser = argparse.ArgumentParser(
        description=BRIEF_INTRO,
        formatter_class=CustomHelpFormatter
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="Paths to code files or directories"
    )
    parser.add_argument(
        "-s", "--linter-script",
        default="./lint.sh",
        help="Path to script that runs code quality checks"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="enable debug logging"
    )
    parser.add_argument(
        "-m", "--model",
        help="what LLM to use",
        default="????"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        help="how many iterations to run",
        default=100
    )
    parser.add_argument(
        "-L", "--local",
        action="store_true",
        help="run a local LLM server using Ollama docker container"
    )
    parser.add_argument(
        "--llm-url",
        default="https://api.groq.com/openai/v1/chat/completions",
        help="URL of the LLM server"
    )
    parser.add_argument(
        "--api-key",
        help="Groq API key",
        default=os.getenv("GROQ_API_KEY", "????")
    )
    parser.add_argument(
        "-g", "--no-git",
        action="store_true",
        help="disable git operations (no branches or commits)"
    )
    args = parser.parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(level)
    handler.setLevel(level)
    if args.model == "????":
        args.model = "llama3" if args.local else "llama-3.3-70b-versatile"
    logger.info("Using model %s", args.model)
    return args


def _extract_filename(line: str) -> str | None:
    """Extract filename from a markdown header line.

    Args:
        line (str): Line to parse

    Returns:
        str | None: Filename if found and valid, None otherwise
    """
    m = re.match(r"^ *###\s+(.*\.py)$", line)
    if not m:
        return None
    fname = m.group(1)
    if "path/to/file.py" in fname:
        return None
    return fname


def _parse_llm_output(output: str) -> tuple[list[ProposedFix], str]:
    """Parse LLM output into explanation and new file versions.

    Expects format:

        ### path/to/file.py

        Explanation of changes...

        ```python
        def foo():
            return "bar"
        ```

    Any text outside the triple-backtick block is treated as explanation.
    """
    proposed_fix: ProposedFix | None = None
    fname: str | None = None
    in_python = False
    fix_list: list[ProposedFix] = []
    explanation: str = ""

    # pylint: disable=no-member
    def maybe_add_fix() -> None:
        if proposed_fix is not None:
            if "No changes needed for this file" in proposed_fix.new_content:
                return
            logger.debug("New version of %s:\n%s",
                         proposed_fix.path,
                         proposed_fix.new_content)
            if proposed_fix.something_changed():
                fix_list.append(proposed_fix)
                logger.debug("Added %s to fix list", proposed_fix.path)

    for line in output.split("\n"):
        # Check for filename header
        new_fname = _extract_filename(line)
        if new_fname:
            maybe_add_fix()
            fname = new_fname
            with open(fname, "r", encoding="utf-8") as f:
                old_content = f.read()
            proposed_fix = ProposedFix(
                path=fname,
                old_content=old_content,
                new_content=""
            )
            continue

        # Handle code blocks
        if line.startswith("```python") and fname:
            in_python = True
            continue
        if line.startswith("```") and proposed_fix:
            in_python = False
            continue

        # Add content to current fix
        if proposed_fix is not None and in_python:
            proposed_fix.new_content += line + "\n"
        else:
            explanation += line + "\n"

    # pylint: enable=no-member
    maybe_add_fix()
    return fix_list, explanation


async def run_all_checks(script: str, *sources: str) -> list[CompletedLinterRun]:
    """Run code quality checks on the specified source files.

    Args:
        script: Path to the linter script to run
        *sources: Variable number of source file paths to check

    Returns:
        List of completed linter runs with results
    """
    results: list[CompletedLinterRun] = []
    for source in sources:
        print(f"$ {script} {source}")
        lrun = LinterRun(path=source, cmd=script)
        results.append(await lrun.run())
    return results


async def stop_existing_ollama(client: docker.DockerClient) -> None:
    """Stop any existing Ollama containers using port 11434.

    Args:
        client: Docker client instance
    """
    try:
        containers = client.containers.list(
            filters={'publish': '11434'}
        )
        for container in containers:
            print(f"Stopping existing container {container.id[:12]}")
            container.stop()
    except APIError as e:
        print(f"Error checking for existing containers: {e}")


async def wait_for_server_ready() -> None:
    """Wait for Ollama server to be ready to accept requests."""
    for _ in range(30):
        try:
            response = requests.get(
                'http://localhost:11434/api/tags',
                timeout=5.0
            )
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            await asyncio.sleep(1)
    raise TimeoutError("Ollama server failed to become ready")


async def pull_ollama_model(model: str) -> None:
    """Pull the specified model using Ollama's API.

    Args:
        model: Name of the model to pull
    """
    logger.debug("Pulling the model %s", model)
    response = requests.post(
        'http://localhost:11434/api/pull',
        headers={'Content-Type': 'application/json'},
        json={'name': model},
        timeout=5*MINUTE
    )
    response.raise_for_status()
    logger.info("Successfully pulled model %s", model)


async def start_ollama_server(model: str) -> str:
    """Start the Ollama server in a container using Docker API.

    Args:
        model: Name of the model to pull

    Returns:
        Container ID of the started server

    Raises:
        docker.errors.APIError: If server fails to start
        requests.exceptions.RequestException: If model pull fails
        TimeoutError: If server doesn't become ready
    """
    client = docker.from_env()

    # Stop any existing Ollama containers
    await stop_existing_ollama(client)

    try:
        # Start new Ollama container
        container = client.containers.run(
            'ollama/ollama',
            command='serve',
            detach=True,
            remove=True,
            ports={'11434/tcp': 11434},
            volumes={
                os.path.expanduser('~/ollama_data'): {
                    'bind': '/root/.ollama',
                    'mode': 'rw'
                }
            }
        )
        cid = container.id if isinstance(container.id, str) else "???"
        print(f"Started container: {cid[:12]}")

        # Wait for server and pull model
        await wait_for_server_ready()
        await pull_ollama_model(model)

        return cid

    except APIError as e:
        print(f"Error starting Ollama container: {e}")
        raise


async def process_iteration(
    files: list[str],
    llm: LLMWrangler,
    linter_script: str
) -> str:
    """Process one iteration of the fix loop."""
    results = await run_all_checks(linter_script, *files)
    if all(issue.code == 0 for issue in results):
        return None

    prompt = generate_prompt(linter_script, results)
    logger.debug("Prompt:\n%s", prompt)

    response = await llm.submit(prompt)
    logger.debug("Response:\n%s", response)
    fixes, explanation = _parse_llm_output(response)
    logger.debug("Fixes:\n%s", fixes)
    logger.debug("Explanation:\n%s", explanation)
    for fix in fixes:
        await fix.apply()
    return explanation


async def main() -> None:
    """Main entry point for the loop runner.

    Parses arguments, sets up LLM and git services, and runs the fix loop
    until either all checks pass or max iterations is reached.
    """
    args = parse_args()
    files = args.sources
    iterations = args.iterations

    git_wrangler = None if args.no_git else GitWrangler.create(path=os.getcwd())
    if git_wrangler:
        git_wrangler.start_branch(
            datetime.now().isoformat()
            .replace(":", "_").replace("-", "_").replace(".", "_")
        )

    platform: Platform = "ollama" if args.local else "groq"
    async with await LLMWrangler.create(platform=platform, model=args.model, api_key=args.api_key) as llm:
        while iterations > 0:
            explanation: str = await process_iteration(
                files,
                llm,
                args.linter_script
            )

            if git_wrangler:
                commit_hash = git_wrangler.commit_fixes(files)
                with open("FIXES.md", "a", encoding="utf-8") as f:
                    f.write(f"\n### Fix Applied (commit: {commit_hash[:8]})\n\n{explanation}\n")
                git_wrangler.commit_log("FIXES.md")
            else:
                with open("FIXES.md", "a", encoding="utf-8") as f:
                    f.write(f"\n### Fix Applied\n\n{explanation}\n")

            iterations -= 1
            if iterations == 0:
                logger.warning("Max iterations reached")
                logger.warning("There are still %d files with remaining issues", len(files))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
