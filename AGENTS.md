# Repository instructions

Use the existing Conda environment `myenv3.13` whenever running Python code,
including scripts, tests, and one-off Python commands. Do not use system Python
or create a separate virtual environment.

Run commands with:

```bash
conda run --no-capture-output -n myenv3.13 python <script.py>
```

Use the same environment for Python package management:

```bash
conda run --no-capture-output -n myenv3.13 python -m pip <arguments>
```

If a dependency is incompatible with this environment's Python version, report
the incompatibility rather than silently switching environments or changing its
Python version.
