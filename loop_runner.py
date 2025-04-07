#!/usr/bin/env python
"""Code Quality Auto-Fix Loop Runner

This script automates the process of fixing code quality issues by:
1. Running code quality checks (pylint, flake8, pytest)
2. Sending the output to an LLM (either Groq API or local Ollama)
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
"""

import os
import sys
import re
import asyncio
import argparse
import jinja2
import requests
from pydantic import BaseModel, ConfigDict
import docker
from docker.errors import NotFound, APIError

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


class LinterRun(BaseModel):
    """Represents a linter run configuration before execution.

    Attributes:
        path (str): Path to the file being linted
        cmd (str): Command used to run the linter
    """
    path: str
    cmd: str
    content: str   # contents of the file being linted

    model_config = ConfigDict(frozen=True)

    def __init__(self, path: str, cmd: str):
        with open(path, "r", encoding="utf-8") as f:
            super().__init__(
                path=path,
                cmd=cmd,
                content=f.read()
            )

    async def run(self) -> 'CompletedLinterRun':
        """Execute the linter command and return a CompletedLinterRun.

        Returns:
            CompletedLinterRun: A new instance with the run results
        """
        proc = await asyncio.create_subprocess_exec(
            self.cmd, self.path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        code = -1 if proc.returncode is None else proc.returncode
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

# pylint: disable=unexpected-keyword-arg
        return CompletedLinterRun(
            path=self.path,
            cmd=self.cmd,
            content=self.content,
            code=code,
            stdout=stdout_str,
            stderr=stderr_str
        )
# pylint: enable=unexpected-keyword-arg


class CompletedLinterRun(LinterRun):
    """Represents a completed linter run with its outputs and status.

    Attributes:
        path (str): Path to the file being linted
        cmd (str): Command used to run the linter
        content (str): Contents of the file being linted
        code (int): Return code from the linter (0 for success)
        stdout (str): Standard output from the linter
        stderr (str): Standard error from the linter
    """
    code: int
    stdout: str
    stderr: str

    model_config = ConfigDict(frozen=True)

    # pylint: disable=non-parent-init-called
    # pylint: disable=super-init-not-called
    def __init__(self, *, path: str, cmd: str, content: str, code: int, stdout: str, stderr: str):
        BaseModel.__init__(
            self,
            path=path,
            cmd=cmd,
            content=content,
            code=code,
            stdout=stdout,
            stderr=stderr
        )
    # pylint: enable=non-parent-init-called
    # pylint: enable=super-init-not-called


class ProposedFix(BaseModel):
    """Represents a fix proposed by the LLM for a code quality issue.

    This class encapsulates a proposed code fix, including the file path,
    the LLM's explanation of the changes, and the new content to be written
    to the file.

    Attributes:
        path (str): Path to the file being modified
        explanation (str): LLM's explanation of what changes were made and why
        new_content (str): Complete new content for the file after fixes
    """
    path: str
    explanation: str
    old_content: str
    new_content: str

    model_config = ConfigDict(populate_by_name=True)

    def __repr__(self) -> str:
        """Return a string representation of the proposed fix.

        Returns:
            str: Format: "<ProposedFix {path}>"
        """
        def short(s: str) -> str:
            n = 40
            if len(s) <= n:
                return s
            return s[:n] + "..."
# pylint: disable=no-member
        return (
            f"<ProposedFix {self.path} "
            f"expl='{short(self.explanation)}' "
            f"new_content='{short(self.new_content)}'>"
        )
# pylint: enable=no-member

    def to_prompt(self) -> str:
        """Return a string representation of the proposed fix
        in the format expected by the LLM.

        Returns:
            str: ....
        """
        width = 80

        def wrap(text: str | None = None, ch: str = '=') -> str:
            if text is None:
                return width * ch
            a = (width - len(text)) // 2 - 1
            b = width - len(text) - a - 2
            return a * ch + " " + text + " " + b * ch

# pylint: disable=no-member
        return (
            wrap(ch='<') + "\n" +
            wrap(self.path) + "\n" +
            self.old_content + "\n" +
            wrap() + "\n" +
            self.explanation + "\n" +
            wrap() + "\n" +
            self.new_content + "\n" +
            wrap(ch='>')
        )
# pylint: enable=no-member

    async def apply(self) -> bool:
        """Apply a a proposed fix.
        """
# pylint: disable=no-member
        path = self.path
        with open(path, "r", encoding="utf-8") as f:
            old_content = f.read()
        if old_content == self.new_content:
            # the LLM didn't change the file, so stop
            # trying to fix it, maybe issue a warning??
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.new_content)
        return True
# pylint: enable=no-member


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
        "--linters",
        default="ruff,pylint,flake8,pytest",
        help="comma-separated list of code quality checkers"
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
    args = parser.parse_args()
    DEBUG = args.debug
    if args.model == "????":
        args.model = "codellama" if args.local else "qwen-2.5-32b"
    return args


def generate_prompt(results: list[CompletedLinterRun]) -> str:
    """Generate a prompt for the LLM

    Args:
        results (list[CompletedLinterRun]): List of code quality issues found by checkers
        files (list[str]): List of file paths to check

    Returns:
        str: Formatted prompt text for the LLM
    """
    lookup = {
        lrun.path: lrun
        for lrun in results
    }
    _files = sorted(lookup.keys())
    if DEBUG > 0:
        print(lookup)
    template = jinja2.Template(PROMPT_TEMPLATE)
    return template.render(files=_files, lookup=lookup)


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


def _parse_llm_output(output: str) -> list[ProposedFix]:
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
        list[ProposedFix]: List of proposed fixes
    """
    proposed_fix = None
    fname = None
    in_python = False
    fix_list = []

    for line in output.split("\n"):
        # Check for filename header
        new_fname = _extract_filename(line)
        if new_fname:
            if proposed_fix is not None:
                fix_list.append(proposed_fix)
            fname = new_fname
            proposed_fix = ProposedFix(
                path=fname,
                explanation=line + "\n",
                old_content=open(fname, "r", encoding="utf-8").read(),
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

# pylint: disable=no-member
        # Add content to current fix
        if proposed_fix:
            if in_python:
                proposed_fix.new_content += line + "\n"
            else:
                proposed_fix.explanation += line + "\n"
# pylint: enable=no-member

    # Add final fix if exists
    if proposed_fix is not None:
        fix_list.append(proposed_fix)

    return fix_list


async def get_llm_response(model: str,
                           prompt: str,
                           api_key: str,
                           local: bool = False) -> list[ProposedFix]:
    """Get a response from either Groq API or local Ollama server.

    Args:
        model (str): Name of the model to use
        prompt (str): The formatted prompt to send
        api_key (str): Groq API key (only used for remote)
        local (bool): Whether to use local Ollama server

    Returns:
        tuple containing:
        - explanation (str): LLM's explanation of changes
        - new versions (list[File]): List of changed file contents
    """
    try:
        if local:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': model, 'prompt': prompt, 'stream': False},
                timeout=20*MINUTE
            )
        else:
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1
                },
                timeout=5*MINUTE
            )
        response.raise_for_status()
        output = response.json()['response'] if local else response.json()['choices'][0]['message']['content']
        if DEBUG > 0:
            print("Output: ***[[[[[[\n" + output + "\n]]]]]]***")
            if DEBUG > 1:
                print("Bailing")
                sys.exit(0)
        return _parse_llm_output(output)
    except requests.exceptions.RequestException as e:
        print(f"Error calling {'Ollama' if local else 'Groq'} API: {e}")
        return []


async def run_all_checks(linters: str, *sources: str) -> list[CompletedLinterRun]:
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
    results: list[CompletedLinterRun] = []
    for source in sources:
        for cmd in linters.split(","):
            if cmd == "pytest" and not source.startswith("test"):
                continue
            print(f"$ {cmd} {source}")
            lrun = LinterRun(path=source, cmd=cmd)
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
    model: str,
    api_key: str,
    local: bool,
    linters: str
) -> list[str] | None:
    """Process one iteration of the fix loop.

    Args:
        files: List of files to process
        model: Name of the LLM model to use
        api_key: API key for Groq
        local: Whether to use local Ollama
        linters: Comma-separated list of linters to run

    Returns:
        List of explanations for each fix, or None if no files were fixed
    """
    # Run checks
    results: list[CompletedLinterRun] = await run_all_checks(linters, *files)
    if all(issue.code == 0 for issue in results):
        return None

    # Generate prompt
    prompt = generate_prompt(results)
    if DEBUG > 0:
        print("Prompt: ***[[[[[[\n" + prompt + "\n]]]]]]***")
        if DEBUG > 2:
            print("Bailing")
            sys.exit(0)

    # Get and apply fixes
    fixes = await get_llm_response(model, prompt, api_key, local)
    if DEBUG > 0:
        print("Response: ***[[[[[[\n" +
              "\n-=-=-=-=-=-\n".join(map(lambda f: f.to_prompt(), fixes)) +
              "\n]]]]]]***")
    for fix in fixes:
        await fix.apply()
    if DEBUG > 0:
        print("Fixes have been applied")

    if len(fixes) == 0:
        print("No files left to fix")
        return None

    return [fix.explanation for fix in fixes]


async def main() -> None:
    """Main entry point"""
    args = parse_args()
    files = args.sources
    iterations = args.iterations
    history = []  # List of explanations
    container = None

    try:
        if args.local:
            client = docker.from_env()
            container_id = await start_ollama_server(args.model)
            container = client.containers.get(container_id)

        while iterations > 0:
            maybe: list[str] | None = await process_iteration(
                files,
                args.model,
                args.api_key,
                args.local,
                args.linters
            )

            if maybe is not None:
                history.extend(maybe)
            else:
                break

            iterations -= 1
            if iterations == 0:
                print("Max iterations reached")
                print(f"There are still {len(files)} files with remaining issues")

    finally:
        print("\n".join(history))
        # Clean up server if it was started
        if container:
            try:
                container.stop()
            except (NotFound, APIError) as e:
                print(f"Error stopping container: {e}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
