# Loop Runner, a LLM-assisted code-fixing tool

`loop_runner.py` is a tool that automates the process of improving Python code
quality by iteratively applying fixes suggested by an LLM. It runs code quality
checks (like pylint, flake8, and pytest), prompts a language model to fix any
identified issues, applies those fixes, and repeats the process until the code
passes all checks or a maximum number of iterations is reached. This can be
particularly useful for automatically addressing style issues, potential bugs,
and improving overall code maintainability.

This is a work in progress.

#### Ideas for improvements

Beyond general bug fixes and improvements:

- [x] Non-local LLM servers e.g. OpenAI, Groq...
- [ ] Non-Python checkers e.g. shellcheck, jsonlint...
- [ ] Parallel checking of multiple source files
- [ ] A run config file maybe?
- [ ] Add support for persisting history to a file
- [ ] Add support for persisting changes to a file

## Prerequisites

Before using `loop_runner.py`, ensure you have the following installed and configured:

*   **Docker**: Required only if using local LLM mode (`-L` flag). Install Docker
    from [https://www.docker.com/](https://www.docker.com/).

*   **Virtual Environment**: It's highly recommended to use a virtual
    environment to manage project dependencies.

    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```
*   **Python Version**: Python 3.8 or higher is required.

    ```
    python3 --version
    ```

    Install the required Python packages using pip:

    ```
    pip install -r requirements.txt
    ```

*   **Groq API Key**: Required if not using local mode. Get your API key from
    [https://console.groq.com](https://console.groq.com) and set it as an environment variable:
    ```
    export GROQ_API_KEY=your_api_key_here
    ```

## Usage

```shell
usage: loop_runner.py [-h] [-d DEBUG] [-m MODEL] [-i ITERATIONS] [-L] [--linters LINTERS] 
                      [--llm-url LLM_URL] [--api-key API_KEY] sources [sources ...]

This script automates the process of fixing code quality issues by:
1. Running code quality checks (e.g. ruff, pylint, flake8, pytest)
2. Asking an LLM to fix any issues found
3. Applying the suggested fixes
4. Repeating until all checks pass

positional arguments:
  sources               Paths to code files or directories

options:
  -h, --help            show this help message and exit
  -d DEBUG, --debug DEBUG
                        how much debug output to produce (default: 0)
  -m MODEL, --model MODEL
                        what LLM to use (default: codellama" if local else "qwen-2.5-32b")
  -i ITERATIONS, --iterations ITERATIONS
                        how many iterations to run (default: 100)
  -L, --local          run a local LLM server using Ollama docker container
  --linters LINTERS     comma-separated list of code quality checkers
                        (default: ruff,pylint,flake8,pytest)
  --llm-url LLM_URL     URL of the LLM server
                        (default: https://api.groq.com/openai/v1/chat/completions)
  --api-key API_KEY     Groq API key (default: $GROQ_API_KEY)
```

## Example

Assume you have a Python file named `example.py` that you want to improve.

1.  **Run `loop_runner.py` on your file using Groq (default):**

    ```
    python loop_runner.py example.py
    ```

    Or using a local Ollama server:

    ```
    python loop_runner.py -L -m codellama example.py
    ```

    This will run the code quality checks, prompt the LLM to fix any issues,
    apply the suggested fixes, and repeat until the code passes all checks or the
    maximum number of iterations is reached.

2.  **View the results:** After the script completes, your `example.py` file
    will be modified with the suggested improvements. You can then review the
    changes and ensure they meet your expectations.

## Model recommendations

The script supports two modes of operation:

### Groq Cloud API (Default)

Groq provides fast inference using their cloud API. The recommended models are:

| Model | Advantages | Disadvantages |
|-------|------------|---------------|
| qwen-2.5-32b (default) | Excellent code understanding; state-of-the-art performance; very fast inference times | Requires API key; usage costs |
| llama-3.3-70b-versatile | Larger context window; more general-purpose capabilities; very fast inference times | Requires API key; usage costs; may be less specialized for code |

### Local Ollama Models (-L flag)

Here's a comparison of Ollama-supported language models suitable for code quality and repair:

| Model           | Advantages                                                                                                                                                                                                                                         | Disadvantages                                                                                             |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| codellama (default) | Specifically fine-tuned for code; excels at code completion, bug detection, and suggesting improvements; variations for different tasks (instruct, python, code).                                                                               | May not be as strong in general language tasks compared to general-purpose models.                             |
| starcoder2      | Designed for code-related tasks with multi-language support; the 15B parameter model is highly performant, matching much larger models on evaluations.                                                                                          | Performance might vary across different programming languages.                                             |
| deepseek-v2     | Optimized for both general text generation and coding tasks.                                                                                                                                                                                      | Might not be as specialized in code-specific tasks as Code Llama or StarCoder2.                                |
| qwen2.5-coder   | State-of-the-art performance among open-source code models, matching GPT-4o in some evaluations; excels in code generation, repair, and reasoning; strong multi-language code repair capabilities; supports context length of up to 128K tokens. | Some evaluations show that the Qwen2.5-Coder-32B-Instruct model may perform comparably to GPT-4o on the Aider benchmark, which may not always be sufficient. |

**General Considerations:**

*   **Model Size vs. Performance:** Larger models generally perform better but require more computational resources.
*   **Task Specialization:** Models fine-tuned for code-specific tasks (like Code Llama and Qwen2.5-Coder) often outperform general-purpose models on coding benchmarks.
*   **Multi-Language Support:** If you work with multiple programming languages, consider models with broad language support (like StarCoder2 and Qwen2.5-Coder).
*   **Resource Constraints:** If you have limited computational resources, opt for smaller models or models designed for efficiency (like Phi-3).

## Convenience Script

The repository includes a convenience script that wraps the python script
and manages the location of the python virtual environment.

```bash
./loop_runner [options] files...
```

You can use all the same options as when running directly:

```bash
# Example with Groq API (default)
./loop_runner -d 1 example.py

# Example with local Ollama
./loop_runner -L -m codellama example.py

# Example with Groq and custom model
./loop_runner --model llama-3.3-70b-versatile example.py
```

You can install `loop_runner` to be runnable from any directory on your
dev machine.

1. Make the script executable:
```bash
chmod +x /path/to/loop_runner.py
```

2. Create a symbolic link in your PATH:
```bash
sudo ln -s /path/to/loop_runner /usr/local/bin/loop_runner
```

3. Set up your virtual environment and add to your .bashrc:
```bash
python -m venv /path/to/your/venv
/path/to/your/venv/bin/pip install -r requirements.txt
echo 'export LOOP_RUNNER_VENV=/path/to/your/venv' >> ~/.bashrc
source ~/.bashrc
```

After this setup, you can run `loop_runner` from anywhere, and it will use the
Python environment specified in `LOOP_RUNNER_VENV`.
