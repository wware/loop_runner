# This will be syntactically correct so it'll pass ruff
# but pylint won't like it. Let's see if loop_runner.py
# can fix that.
def add(x: int, y: int) -> int:
    return x + y
def multiply(x: int, y: int) -> int:
    return x * y
def dotProduct(w: int, x: int, y: int, z: int) -> int:
    return add(multiply(w, y), multiply(x, z))
