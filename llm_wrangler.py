#!/usr/bin/env python
"""LLM Wrangler Service

Handles communication with various LLM platforms (Groq, Ollama, etc).
"""

import asyncio
import os
import logging
from types import TracebackType
from typing import Literal
import aiohttp
import requests
import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container

Platform = Literal["groq", "ollama"]
logger = logging.getLogger("loop_runner")


class LLMWrangler:
    """Service for handling LLM interactions across different platforms."""

    def __init__(self) -> None:
        self.platform: Platform | None = None
        self.model: str | None = None
        self.api_key: str | None = None
        self.container: Container | None = None

    @classmethod
    async def create(cls, platform: Platform, model: str, api_key: str | None = None) -> "LLMWrangler":
        """Create and configure a new LLMWrangler instance.

        Args:
            platform: Which LLM platform to use ("groq" or "ollama")
            model: Name of the model to use
            api_key: API key for cloud services (required for Groq)

        Returns:
            Configured LLMWrangler instance
        """
        wrangler = cls()
        await wrangler._configure(platform, model, api_key)
        return wrangler

    async def _configure(self, platform: Platform, model: str, api_key: str | None = None) -> None:
        """Internal configuration method."""
        self.platform = platform
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if platform == "ollama":
            await self._start_ollama()

    async def submit(self, prompt: str) -> str:
        """Submit a prompt to the configured LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response text

        Raises:
            RuntimeError: If LLM is not configured
        """
        if not self.platform or not self.model:
            raise RuntimeError("LLM not configured. Call configure() first.")

        async with aiohttp.ClientSession() as session:
            try:
                if self.platform == "ollama":
                    async with session.post(
                        'http://localhost:11434/api/generate',
                        json={'model': self.model, 'prompt': prompt, 'stream': False},
                        timeout=aiohttp.ClientTimeout(total=20*60)  # 20 minutes
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data['response']

                elif self.platform == "groq":
                    if not self.api_key:
                        raise RuntimeError("Groq API key required")

                    async with session.post(
                        'https://api.groq.com/openai/v1/chat/completions',
                        headers={
                            'Authorization': f'Bearer {self.api_key}',
                            'Content-Type': 'application/json'
                        },
                        json={
                            'model': self.model,
                            'messages': [{'role': 'user', 'content': prompt}],
                            'temperature': 0.1
                        },
                        timeout=aiohttp.ClientTimeout(total=5*60)  # 5 minutes
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data['choices'][0]['message']['content']

            except aiohttp.ClientError as e:
                print(f"Error calling {self.platform} API: {e}")
                return ""

    async def _start_ollama(self) -> None:
        """Start Ollama container if using local mode."""
        client = docker.from_env()

        # Remove existing container if it exists
        try:
            container = client.containers.get("ollama")
            logger.debug("Removing existing Ollama container")
            container.remove(force=True)
        except NotFound:
            pass

        # Start fresh container
        logger.debug("Starting new Ollama container")
        self.container = client.containers.run(
            "ollama/ollama",
            name="ollama",
            detach=True,
            ports={'11434/tcp': 11434},
            volumes={'/root/ollama_data': {'bind': '/root/.ollama', 'mode': 'rw'}}
        )

        # Wait for server to be ready
        for _ in range(30):
            try:
                requests.get('http://localhost:11434/api/tags',
                             timeout=10)
                break
            except requests.exceptions.RequestException:
                await asyncio.sleep(1)

        # Pull model if needed
        response = requests.post(
            'http://localhost:11434/api/pull',
            json={'name': self.model},
            timeout=3600
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to pull model {self.model}")

    async def __aenter__(self) -> "LLMWrangler":
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc_val: BaseException | None,
                        exc_tb: TracebackType | None) -> None:
        if self.container:
            try:
                self.container.stop()
            except (NotFound, APIError) as e:
                logger.error("Error stopping container: %s", e)
