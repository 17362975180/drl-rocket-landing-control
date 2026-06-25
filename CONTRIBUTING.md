# Contributing

Thanks for your interest in this project.

## Development Setup

Use Python 3.10 or 3.11.

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python smoke_tests.py
```

On macOS/Linux, create a virtual environment manually and install the same
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_tests.py
```

## Pull Request Checklist

- Keep generated experiment outputs out of commits unless they are small
  summaries needed for reproducibility.
- Run `python smoke_tests.py` before opening a pull request.
- If model behavior changes, document the command, seed, and evaluation
  protocol used to produce the result.
- Do not commit local virtual environments, scratch files, rendered resumes, or
  private documents.
