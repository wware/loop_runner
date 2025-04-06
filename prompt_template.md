# Please fix the following issues in some Python source files

## Source files

{% for file in files %}
### {{ file.name }}

```python
{{ file.content }}
```
{% if file.issue %}

#### issue(s) with this file:
{{ file.issue.content }}
{% endif %}
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
