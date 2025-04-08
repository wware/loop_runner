"""Data models for the Loop Runner."""

import logging
import asyncio
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("loop_runner")


class LinterRun(BaseModel):
    """Represents a linter run configuration before execution."""
    path: str
    cmd: str
    content: str    # file content of the file to be linted

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
        stdout, stderr = await proc.communicate(self.content.encode())
        code = -1 if proc.returncode is None else proc.returncode

        return CompletedLinterRun(
            path=self.path,
            cmd=self.cmd,
            content=self.content,
            code=code,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )


class CompletedLinterRun(BaseModel):
    """Represents a completed linter run with its outputs and status."""
    path: str
    cmd: str
    content: str
    code: int
    stdout: str
    stderr: str

    model_config = ConfigDict(frozen=True)


class ProposedFix(BaseModel):
    """Represents a fix proposed by the LLM for a code quality issue.

    This class encapsulates a proposed code fix, including the file path
    and the new content to be written to the file.

    Attributes:
        path (str): Path to the file being modified
        new_content (str): Complete new content for the file after fixes
    """
    path: str = Field(frozen=True)
    old_content: str = Field(frozen=True)
    new_content: str

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
        return (
            f"<ProposedFix {self.path} "
            f"new_content='{short(self.new_content)}'>"
        )

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

        return (
            wrap(ch='<') + "\n" +
            wrap(self.path) + "\n" +
            self.old_content + "\n" +
            wrap() + "\n" +
            self.new_content + "\n" +
            wrap(ch='>')
        )

    def something_changed(self) -> bool:
        """Check if the proposed fix has changes.

        Returns:
            bool: True if the content changed, False otherwise
        """
        return self.old_content != self.new_content

    async def apply(self) -> bool:
        """Apply a proposed fix.

        Returns:
            bool: True if changes were applied, False if no changes needed
        """
        path = self.path
        with open(path, "r", encoding="utf-8") as f:
            old_content = f.read()
        if old_content == self.new_content:
            logger.debug("No changes to %s", self.path)
            # the LLM didn't change the file, so stop
            # trying to fix it, maybe issue a warning??
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.new_content)
        logger.debug("Wrote %s", self.path)
        return True
