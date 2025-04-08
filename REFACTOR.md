# Refactor

## Where things are right now

The repository, named
[**Loop Runner**](https://github.com/wware/loop_runner),
is a tool designed to automate the
process of improving Python code quality using an iterative approach with
a large language model (LLM). It performs code quality checks (e.g.,
`pylint`, `flake8`, `pytest`), prompts an LLM to fix identified issues,
applies those fixes, and repeats until the code passes all checks or
reaches a maximum number of iterations.

### Key Features of Loop Runner:
- **Purpose:** Automates fixing style issues, potential bugs, and
  improving code maintainability.
- **Supported Tools:** Works with Python code quality checkers like
  `ruff`, `pylint`, `flake8`, and `pytest`.
- **LLM Integration:** Supports both cloud-based LLMs (e.g., Groq API)
  and local LLM servers (e.g., Code Llama via Docker).
- **Customization Options:** Allows specifying models, iterations,
  linters, and more through command-line arguments.

### Prerequisites:
- Python 3.8 or higher.
- Virtual environment setup for dependencies.
- Optional: Docker for local LLM usage or a Groq API key for cloud-based LLMs.

### Usage Example:
To improve a file named `example.py`:
1. Using Groq API (default):
   ```bash
   python loop_runner.py example.py
   ```
2. Using a local server:
   ```bash
   python loop_runner.py -L -m codellama example.py
   ```

The script iteratively modifies the file with suggested improvements until all checks pass or the iteration limit is reached.

## Future plans

> I'm thinking about generalizing this by splitting out the linters as a
> separate shell script, and then split out the modification stage as another
> shell script. I have a hunch that this would allow the framework to be used
> for other purposes that might turn out useful. And it might also improve
> clarity and maintainability. Any thoughts?

Your idea to split the linters and modification stages into separate shell scripts
is a thoughtful approach that could enhance the flexibility, clarity, and
maintainability of the framework. Here are some considerations and potential benefits:

### **Benefits of Splitting Linters and Modifications:**

1. **Improved Modularity:**
   - By separating the linters into their own shell script, you can make it
     easier to add or remove linters without modifying the core logic of the framework.
   - Similarly, isolating the modification stage allows you to swap out or update
     the logic for code improvements independently.

2. **Framework Generalization:**
   - This separation could make the framework adaptable to other use cases
     beyond Python code quality improvement. For example:
   - Running linters for other programming languages.
   - Performing static analysis or security checks.
   - Automating other iterative workflows (e.g., data preprocessing or
     document formatting).

3. **Enhanced Clarity:**
   - Keeping each stage in a separate script makes the workflow easier to
     understand and debug. Each script would have a single responsibility,
     reducing complexity.

4. **Ease of Maintenance:**
   - Changes to linters or modification logic can be made without affecting unrelated parts of the framework.
   - Shell scripts are lightweight and easy to test independently.

5. **Cross-Language Compatibility:**
   - If the linter script is generalized, it could support multiple languages
     by detecting file types or accepting language-specific configurations
     as arguments.

### **Implementation Suggestions:**

1. **Linter Script (`run_linters.sh`):**
   - This script could take a file path as input and run all configured
     linters sequentially, outputting results in a standardized format (e.g.,
     JSON or plain text).
   - Example:
     ```bash
     #!/bin/bash
     file=$1
     pylint $file
     flake8 $file
     ruff $file
     ```

2. **Modification Script (`apply_fixes.sh`):**
   - This script could accept linter output and invoke the LLM or other tools
     to modify the code based on detected issues.
   - Example:
     ```bash
     #!/bin/bash
     file=$1
     issues=$2  # Path to linter output file
     python modify_code.py --file $file --issues $issues
     ```

3. **Main Controller (`loop_runner.sh`):**
   - The main script would orchestrate the process:
     1. Run `run_linters.sh`.
     2. Pass linter results to `apply_fixes.sh`.
     3. Repeat until success or iteration limit is reached.

4. **Extensibility:**
   - Use environment variables or configuration files for customizable settings
     (e.g., which linters to run, iteration limits).
   - Allow users to plug in additional tools by modifying only the shell scripts.

### **Potential Challenges:**

1. **Inter-Script Communication:**
   - Ensure that outputs from one script are properly formatted for input into
     another (e.g., using JSON or structured logs).
   
2. **Dependency Management:**
   - Shell scripts may require additional setup for dependencies (e.g., installing
     linters). Consider automating this process within the scripts.

3. **Performance Considerations:**
   - Running separate scripts may introduce overhead compared to a single Python
     script managing everything internally. However, this trade-off is often worth
     it for modularity.

### **Conclusion:**
Your approach aligns well with principles of software engineering, such as
modularity and separation of concerns. Splitting out these stages will make
your framework more versatile and easier to maintain while opening up
possibilities for broader applications beyond code quality improvement.
It’s worth pursuing!

## Simpler prompt format

> When listing the source files, you want to use a very straightforward format
> that every LLM will understand. So maybe present a prompt to the LLM that
> looks like this.
> 
> ```
> Here are some source files we will be working with
> <<<< foo.py <<<<
> def do_some_stuff(*args):
>     here is the stuff we are doing
> def another_function():
>     so much code, such pretty code
> >>>>>>>>
> <<<< bar.py <<<<
> class StillMoreGreatPythonStuff:
>     def method1(...):
>         xxxx
>     def method2(...):
>         xxxx
> >>>>>>>>>
> We will process these with the following script
> <<<< my_linter_script.sh <<<<
> #!/bin/bash -xe
> do stuff
> do more stuff
> produce output that tells us important things
> >>>>>>>>
> This script is applied with the following command line
> <<<<<<<<
> ./my_linter_script.sh --fee --fi --fo --fum foo.py bar.py
> >>>>>>>>
> yielding the following, with exit status 1
> <<<< stdout <<<<
> bunches of useful info generated by script
> >>>>>>>>
> <<<< stderr <<<<
> >>>>>>>>
> Given all the preceding, here are instructions about what to do next
> <<<< instructions <<<<
> do stuff
> do more stuff
> based on stuff up to now, we might generate new versions of the source files
> or do something else entirely, we can make that decision later
> >>>>>>>>
> ```
> 
> So it's more of a plain-text format with very clear distinct pieces,
> hopefully leaving maximum flexibility for adapting what this thing can do,
> by tweaking the shell script, its command line, and the instructions at the end.
> 
> The parts that stay the same are the LLM wrangling (Groq vs Ollama,
> choice of model) and the idea that you're going to iterate a loop until
> some condition is met, or you've hit a maximum number of iterations. The
> prompt can still be constructed with Jinja-style templating.
> 
> Something to add is that at the beginning it should create a new git branch,
> and after each apply-fix step it should do a git commit, so that the entire
> history is preserved. Any explanations should go into a file with a name like
> `FIXES.md`.

Your approach of creating a highly structured prompt is excellent. It maximizes
flexibility and clarity, which are crucial for adapting the framework to various
tasks. Let's break down the components and see how they fit together.

### **Advantages of This Structured Prompt Format:**

1. **Maximum Flexibility:**
   - By including the linter script, its command line, and the instructions at
     the end, you provide ample room for customization without changing the
     core logic of the framework.
   - This allows you to adapt to different tasks simply by tweaking these elements.

2. **Clear Separation of Concerns:**
   - Each section (source files, linter script, command line, stdout, stderr,
     instructions) is clearly delineated, making it easy for the LLM to
     understand the context and purpose.

3. **Complete Context:**
   - The LLM has access to everything it needs to make informed decisions: source
     code, how it's being processed, the output of that processing, and specific
     instructions on what to do next.

4. **Human-Readable:**
   - The prompt format is easy for humans to read and understand, which aids in
     debugging and fine-tuning.

### **Additional Features to Integrate:**

1. **Git Branching and Commits:**
   - At the beginning of the process, create a new Git branch:
     ```bash
     git checkout -b my-new-branch
     ```
   - After each "apply-fix" step, commit the changes with a meaningful message:
     ```bash
     git add .
     git commit -m "Apply fix: [description of changes]"
     ```
   - This ensures that the entire history of changes is preserved and easily reviewable.

2. **FIXES.md File:**
   - Create a file named `FIXES.md` to document the changes made during each iteration.
   - Each entry should include:
     - Commit hash.
     - Description of the changes.
     - Rationale for the changes.
   - Example:
     ```markdown
     ### Commit [commit hash]
     - Applied fix to address Pylint error E1101 in `foo.py`.
     - Rationale: The error was due to accessing a non-existent attribute, which has now
       been corrected.
     ```

3. **Jinja-Style Templating:**
   - Use Jinja templates to construct the prompt dynamically. This allows you to
     easily insert file contents, script outputs, and other variables into the prompt.
   - Example:
     ```python
     from jinja2 import Template
     template_string = """
     Here are some source files:
     {% for file, content in source_files.items() %}
     >>>>>>>
     {% endfor %}
     """
     template = Template(template_string)
     rendered_prompt = template.render(source_files={"foo.py": "...", "bar.py": "..."})
     ```

4. **Loop Control:**
   - Implement a loop that continues until a certain condition is met (e.g., all linter
     checks pass) or a maximum number of iterations is reached.
   - The loop should:
     1. Construct the prompt.
     2. Send the prompt to the LLM.
     3. Apply the changes suggested by the LLM.
     4. Commit the changes to Git.
     5. Update the `FIXES.md` file.

### **Implementation Steps:**

1. **Setup:**
   - Create a new Git branch.
   - Initialize the `FIXES.md` file.

2. **Loop:**
   - Construct the prompt using Jinja templates.
   - Send the prompt to the LLM.
   - Apply the changes suggested by the LLM.
   - Commit the changes to Git.
   - Update the `FIXES.md` file with details of the changes.
   - Run linters to check if the condition is met.
   - Repeat until the condition is met or the maximum number of iterations is reached.

### **Advantages of This Implementation:**

1. **Preservation of History:**
   - Every change is tracked in Git, providing a complete history of the modifications
     made by the LLM.
   - The `FIXES.md` file provides additional context and rationale for each change.

2. **Reproducibility:**
   - The entire process can be easily reproduced by checking out the Git branch and
     following the steps outlined in the `FIXES.md` file.

3. **Flexibility:**
   - The prompt format allows you to easily adapt to different tasks and programming languages.

### **Conclusion:**

Your approach of using a structured prompt with Jinja templating is an excellent way
to maximize flexibility and clarity. Integrating Git branching and a `FIXES.md` file
will further enhance the framework by preserving the history of changes and providing
additional context. This approach is well-suited for creating a versatile and adaptable
framework that can be used for a wide range of tasks.

## More great stuff

> One of the tricky things here is to know when it's done. Maybe the instructions
> should ask the LLM to declare "No further improvements possible" or some such.
> Or the absence of any new versions of files, or all the new versions are
> identical to the current versions.
> 
> LLM-wrangling might be split out as a distinct service. So that's one separate
> piece, hopefully testable because it's only handling that one concern. I would
> maybe add other non-local LLMs like Perplexity or Claude. (Not a big fan of
> ChatGPT these days.)
> 
> That leaves the looper who is now concerned with
> 
> ```
> start a new git branch
> while True:
>     running the linter (or whatever) script
>     generating the prompt
>     submitting the prompt
>     fetching LLM response and parsing it
>     modifying source files (conditionally)
>     git commit of file mods (conditionally)
>     adding explanation to FIXES.md
>     git commit of FIXES.md (includes git hash for file-mod commit)
>     if no source files were modded:
>         break
> ```
> 
> I'm trying to see if there are any other major chunks that can be split out as
> separate things like the LLM wrangler. Not seeing any other big splittable
> chunks at the moment.
> 
> There should be some fixed instructions about how to format new versions of
> source files, so that parsing them can be consistent, and independent of any
> flexible instructions. Anything that isn't new-version stuff gets put into
> FIXES.md.

Your refined approach to determine when the loop is complete and to split out
the LLM wrangler as a distinct service is well-considered. Let's analyze these
improvements and see if there are any other major chunks that can be separated.

### **Determining Loop Completion:**

1. **LLM Declaration:**
   - Asking the LLM to explicitly declare "No further improvements possible" is
     a robust method. It puts the decision directly in the hands of the model,
     which has seen all the code and linter outputs.

2. **Absence of New Versions:**
   - If the LLM doesn't provide new versions of files, that's a clear signal
     that it couldn't find anything to improve.

3. **Identical New Versions:**
   - If the new versions of the files are identical to the current versions, it
     means the LLM is not making any meaningful changes.

By combining these three signals, you can create a reliable mechanism to break
out of the loop.

### **LLM Wrangler as a Distinct Service:**

1. **Purpose:**
   - The LLM wrangler's sole responsibility is to handle communication with
     various language models (Groq, Ollama, Perplexity, Claude, etc.).

2. **Testability:**
   - By isolating this concern, you can easily test the LLM wrangler with
     different prompts and models to ensure it's working correctly.

3. **Flexibility:**
   - You can add new language models or modify the communication logic without
     affecting the rest of the framework.

### **Potential Additional Splits:**

1. **Prompt Generation:**
   - While the looper orchestrates the process, the actual generation of the
     prompt could be split out into a separate module or function.
   - This would encapsulate the logic for formatting the prompt, inserting file
     contents, and adding instructions.

2. **Git Interaction:**
   - All Git-related operations (creating branches, committing changes) could
     be abstracted into a separate Git service or module.
   - This would allow you to easily switch to a different version control
     system in the future if needed.
   - https://gitpython.readthedocs.io/en/stable/tutorial.html
   - `pip install GitPython==0.3.2`

3. **File Modification and Parsing:**
   - Isolate the file modification and parsing tasks into a dedicated module.
   - Standardize the way new versions of files are handled and parsed, ensuring
     consistency across iterations.

4. **Reporting/FIXES.md Handling:**
   - Encapsulate all logic related to generating and updating the `FIXES.md`
     file into a dedicated module.
   - This could include formatting commit messages, adding explanations, and
     maintaining a consistent structure.

### **Fixed Instructions for New Versions of Source Files:**

1. **Format:**
   - New versions of source files should be clearly demarcated within the LLM's
     response.

2. **Parsing:**
   - The looper should have a consistent way to parse these new versions,
     regardless of the specific instructions given to the LLM.

3. **Handling Unchanged Files:**
   - If a file hasn't changed, the LLM should explicitly indicate this (e.g.,
     by including a special marker or omitting the file altogether).

### **Conclusion:**

By implementing these improvements and potential splits, you're creating a
highly modular and adaptable framework. The key is to isolate concerns and
create well-defined interfaces between different components. This will make the
framework easier to test, maintain, and extend in the future.

## LLM wrangler

The LLM wrangler service will manage interactions with various language model
platforms. It provides function calls for configuring the service, submitting
prompts, and retrieving responses.

### Example usage

```python
from llm_wrangler import LLMWrangler

async with LLMWrangler.create(platform="ollama", model="codellama") as llm_wrangler:
    prompt = """
    some big long prompt...
    """
    response: str = await llm_wrangler.submit(prompt)
```
