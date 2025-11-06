Reminder: The README should allow anyone to jump into the project within a few minutes. Keeping it updated is an ABSOLUTE PRIORITY.

# PRC Challenge 2025

This repository contains the codebase of Euranova's contribution to Eurocontrol's 2025 challenge.

## Getting started as a contributor

Use uv.
If you need the explicit creation of a virtual environment, for example to run notebooks in VS Code, run within the directory which contains this README:
```bash
uv venv
```
You can now select the environment related to the project in VS Code.

#### Accessing the data

```bash
export ACCESS_KEY=xxxxxxxxxxxxxxxx
export SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
bash load_data.sh
```

### Adding a new dependency

```bash
uv add package_name
```

If everything goes right, you'll just have to make other devs aware that they need to pull and check that what they work on is still working.

If uv can't manage to find a proper version of the package because of current dependencies, see with the others if the blocking packages requirements can be relaxed.

# Architecture of the repo

