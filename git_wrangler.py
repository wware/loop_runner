"""Git operations wrapper for the Loop Runner.

Handles git operations like creating branches and committing changes.
"""

import logging
from git import Repo

logger = logging.getLogger("loop_runner")


class GitWrangler:
    """Handles git operations for the loop runner."""

    def __init__(self, repo: Repo):
        self.repo = repo
        self.branch: str | None = None

    @classmethod
    def create(cls, path: str) -> "GitWrangler":
        """Create a GitWrangler for the repo at the given path.

        Args:
            path: Path to git repository

        Returns:
            Configured GitWrangler instance
        """
        repo = Repo(path)
        return cls(repo)

    def start_branch(self, branch_name: str) -> None:
        """Create and switch to a new branch.

        Args:
            branch_name: Name for the new branch
        """
        self.branch = f"autofix/{branch_name}"
        current = self.repo.active_branch
        self.repo.git.checkout('-b', self.branch)
        logger.info("Created branch %s from %s",
                    self.branch, current)

    def commit_fixes(self, files: list[str]) -> str:
        """Commit the fixed files.

        Args:
            files: List of files to commit

        Returns:
            The commit hash
        """
        self.repo.index.add(files)
        commit = self.repo.index.commit("Auto-fixes from LLM")
        return commit.hexsha

    def commit_log(self, fixes_md: str) -> str:
        """Commit the FIXES.md file.

        Args:
            fixes_md: Path to FIXES.md file

        Returns:
            The commit hash
        """
        self.repo.index.add([fixes_md])
        commit = self.repo.index.commit("Update fix history")
        return commit.hexsha
