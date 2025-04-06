#!/usr/bin/env python
"""Code Quality Auto-Fix Loop Runner

This script automates the process of fixing code quality issues by:
1. Running code quality checks (pylint, flake8, pytest)
2. Sending the output to an LLM (running on Ollama)
3. Applying the suggested fixes
4. Repeating until all checks pass

The script uses Docker to run the LLM server, ensuring consistent
access to the model without local dependencies. Code checks are
run directly on the host machine.

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
"""

import os
import re
import subprocess
import sys
import asyncio
import argparse
from typing import Optional
import jinja2
import requests
from pydantic import BaseModel

MINUTE: float = 60.0
DEBUG: int = 0


BRIEF_INTRO = """\
This script automates the process of fixing code quality issues by:
1. Running code quality checks (e.g. ruff, pylint, flake8, pytest)
2. Asking an LLM (running on Ollama) to fix any issues found
3. Applying the suggested fixes
4. Repeating until all checks pass
The LLM runs in a Docker container. Code checks and patching are
done directly on the host machine.
"""


class Issue(BaseModel):
    """Represents a code quality issue found by a checker

    Attributes:
        filename: Path to the file containing the issue
        code: Exit status of the check (0 for success)
        content: Combined stdout/stderr from the check
    """
    filename: str
    code: int
    content: str

    def to_dict(self) -> dict:
        """Convert the issue to a dictionary for serialization"""
        return {
            "filename": self.filename,
            "content": self.content
        }


class File(BaseModel):
    """Represents a source file to be checked and potentially modified

    Attributes:
        name: Path to the file
        content: Current content of the file, read from disk if not provided
    """
    name: str
    content: str = ""
    issue: Optional[Issue] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.content:
            with open(self.name, "r", encoding="utf-8") as f:
                self.content = f.read()

    def to_dict(self) -> dict:
        """Convert the file to a dictionary for serialization"""
        return {
            "name": self.filename,
            "content": self.content
        }


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
                            model name, iterations, linters, and llm-url
    """
    global DEBUG
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
        "-d", "--debug",
        type=int,
        default=0,
        help="how much debug output to produce"
    )
    parser.add_argument(
        "-m", "--model",
        help="what LLM to use",
        default="codellama"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        help="how many iterations to run",
        default=100
    )
    parser.add_argument(
        "--linters",
        default="ruff,pylint,flake8,pytest",
        help="comma-separated list of code quality checkers"
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:11434/v1/chat/completions",
        help="URL of the LLM server"
    )
    args = parser.parse_args()
    DEBUG = args.debug
    return args


def generate_prompt(results: list[Issue],
                    files: list[str]) -> str:
    """Generate a prompt for the LLM

    Args:
        results (list[Issue]): List of code quality issues found by checkers
        files (list[str]): List of file paths to check

    Returns:
        str: Formatted prompt text for the LLM
    """
    _files = []
    lookup = {
        issue.filename: issue
        for issue in results
    }
    if DEBUG > 0:
        print(lookup)
    _files = [
        File(name=f, issue=lookup.get(f, None))
        for f in files
    ]
    assert any(map(lambda f: f.issue is not None, _files)), _files
    with open("prompt_template.md", "r", encoding="utf-8") as f:
        template = jinja2.Template(f.read())
        return template.render(files=_files)


def _parse_llm_output(output: str) -> tuple[str, list[File]]:
    """Parse LLM output into explanation and new file versions.

    Parses Markdown-formatted LLM output that contains new file versions.
    Expects format:
    ### filename.py
    ```python
    # Used to be bad, now it's good
    ```
    Any text outside these blocks is treated as explanation.

    Args:
        output (str): Raw LLM response text

    Returns:
        tuple[str, list[File]]: Tuple containing:
            - str: Text outside of file blocks (explanation)
            - list[File]: List of new file contents
    """
    explanation = ""
    new_version = None
    fname = None
    files = []
    for line in output.split("\n"):
        m = re.match(r"^ *###\s+(.*\.py)$", line)
        if m:
            fname = m.group(1)
            if "path/to/file.py" in fname:
                # Bad filename
                fname = None
            explanation += line + "\n"
        elif line.startswith("```python") and isinstance(fname, str):
            assert new_version is None
            new_version = File(name=fname)
            new_version.content = ""
        elif line.startswith("```") and new_version is not None:
            # Handle end of new version
            files.append(new_version)
            new_version = None
        elif new_version is not None:
            new_version.content += line + "\n"
        else:
            explanation += line + "\n"
    return explanation, files


async def get_llm_response(model: str, prompt: str) -> tuple[str, list[File]]:
    """Get a response from the LLM via Ollama's HTTP API.

    Sends the prompt to the locally running Ollama server and parses the response
    into an explanation and a list of diffs. The LLM is expected to format
    its response in Markdown with file diffs.

    Args:
        model (str): Name of the Ollama model to use
        prompt (str): The formatted prompt to send

    Returns:
        tuple containing:
        - explanation (str): LLM's explanation of changes
        - new versions (list[File]): List of changed file contents

    Raises:
        requests.exceptions.RequestException: If API call fails
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=5*MINUTE
        )
        response.raise_for_status()
        output = response.json()['response']
        if DEBUG > 0:
            print(output)
        return _parse_llm_output(output)
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "", []


async def run_check(cmd: str, source: str) -> Issue:
    """Run a code quality check command locally.

    Executes the command on the host machine and captures output.

    Args:
        cmd (str): Command to run (e.g. "pylint" or "ruff check --fix")
        source (str): Path to the file to check

    Returns:
        Issue: Object containing:
            - filename: Path to the file containing the issue
            - code: Exit status of the check (0 for success)
            - content: Combined stdout/stderr output
    """
    proc = await asyncio.create_subprocess_exec(
        *(cmd + " " + source).split(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    output, error = await proc.communicate()
    if DEBUG:
        print("OUT:" + output.decode())
        print("ERR:" + error.decode())
        if DEBUG > 2:
            sys.exit(0)
    return Issue(filename=source,
                 code=proc.returncode,
                 content=output.decode() + error.decode())


async def run_all_checks(linters: str, *sources: str) -> list[Issue]:
    """Run all code quality checks on the specified source files.

    Executes linters and tests in sequence, stopping at first failure.

    Args:
        linters (str): comma-separated list of linters to run
                       (e.g. "ruff,pylint,flake8,pytest")
        *sources: Variable number of source file paths or directories
                  to check

    Returns:
        List of issues, each containing:
        - filename (str): Path to the file containing the issue
        - code (int): Exit status of the check (0 for success)
        - content (str): Combined stdout/stderr from the check

    Note:
        Checks are run in order and stop at first failure to avoid
        overwhelming the LLM with multiple types of errors at once.
    """
    results: list[Issue] = []
    for source in sources:
        for cmd in linters.split(","):
            if cmd == "ruff":
                cmd = "ruff check"
            elif cmd == "pytest" and not source.startswith("test"):
                continue
            print(f"$ {cmd} {source}")
            issue = await run_check(cmd, source)
            print(f"Exit status {issue.code}")
            if issue.code != 0:
                # only one issue per file for now
                results.append(issue)
                break
    return results


async def start_ollama_server(model: str) -> str:
    """Start the Ollama server in a container.

    Launches a Docker container running the Ollama server which provides
    LLM capabilities via HTTP API. The server:
    - Runs in detached mode (-d flag)
    - Maps the host's ~/ollama_data to container's model storage
    - Exposes port 11434 for API access
    - Auto-removes container when stopped (--rm flag)

    After starting the server, pulls the requested model to ensure
    it's available for use. The pull operation may take several
    minutes for large models being downloaded for the first time.

    Args:
        model (str): Name of the model to pull (e.g. "llama2", "qwen2.5-coder")

    Returns:
        str: Container ID of the started server, used for cleanup

    Raises:
        RuntimeError: If port 11434 is already in use
        subprocess.SubprocessError: If server fails to start
        TimeoutError: If server doesn't respond within 30 seconds
        requests.exceptions.RequestException: If model pull fails
        IOError: If volume mount paths don't exist
    """
    # Kill any container using port 11434
    cmd = ["docker", "ps", "-q", "--filter", "publish=11434"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE
    )
    output, _ = await proc.communicate()
    if output:
        container_id = output.decode().strip()
        print(f"Stopping existing container {container_id}")
        stop_cmd = ["docker", "stop", container_id]
        proc = await asyncio.create_subprocess_exec(*stop_cmd)
        await proc.wait()

    cmd = [
        'docker', 'run', '--rm', '-d',
        '-v', f"{os.path.expanduser('~/ollama_data')}:/root/.ollama",
        '-p', '11434:11434',
        'ollama/ollama', 'serve'
    ]
    if DEBUG:
        print("Starting Ollama server with command:")
        print(" ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    output, error = await proc.communicate()
    if error:
        raise subprocess.SubprocessError(f"Error starting Ollama server: {error.decode()}")

    container_id = output.decode().strip()
    print(f"Started container: {container_id}")

    # Check container status
    status_cmd = ['docker', 'inspect', container_id]
    proc = await asyncio.create_subprocess_exec(
        *status_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    # Wait for server to be ready
    for _ in range(30):
        try:
            response = requests.get(
                'http://localhost:11434/api/tags',
                timeout=5.0
            )
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            await asyncio.sleep(1)

    if DEBUG:
        print("Pulling the model")
    response = requests.post(
        'http://localhost:11434/api/pull',
        headers={'Content-Type': 'application/json'},
        json={'name': model},
        timeout=5*MINUTE
    )
    response.raise_for_status()
    print(f"Successfully pulled model {model}")
    return container_id


async def main():
    """Main entry point"""
    args = parse_args()
    files = args.sources
    iterations = args.iterations
    history = []  # List of explanations

    container_id = await start_ollama_server(args.model)
    try:
        while True:
            # Run checks
            results: list[Issue] = await run_all_checks(args.linters, *files)
            if all(issue.code == 0 for issue in results):
                break
            # Generate prompt
            prompt = generate_prompt(results, files)
            if DEBUG > 0:
                print("Prompt: ***[[[[[[\n" + prompt + "\n]]]]]]***")
            explanation, noobs = await get_llm_response(args.model, prompt)
            history.append(explanation)
            for file in noobs:
                old_content = open(file.name, "r", encoding="utf-8").read()
                if old_content == file.content:
                    # the LLM didn't change the file, so stop trying to fix it
                    files.remove(file.name)
                    continue
                with open(file.name, "w", encoding="utf-8") as f:
                    f.write(file.content)
            if len(files) == 0:
                print("No files left to fix")
                break
            iterations -= 1
            if iterations == 0:
                print("Max iterations reached")
                if len(noobs) > 0:
                    print(f"There are still {len(noobs)} files with remaining issues")
                break
    finally:
        print("\n".join(history))
        # Clean up server at end
        if container_id:
            cmd = ['docker', 'stop', container_id]
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
