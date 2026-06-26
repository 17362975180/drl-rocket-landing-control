# Open Source Release Notes

This document records the checks used to prepare the project for a public GitHub
release.

## Pre-Publish Checklist

- License: MIT license is present in `LICENSE`.
- Citation metadata: `CITATION.cff` is present.
- Community files: issue templates, pull request template, `CONTRIBUTING.md`,
  and `SECURITY.md` are present.
- Ignore rules: virtual environments, local scratch folders, credentials,
  TensorBoard logs, regenerated experiment folders, and ad-hoc model artifacts
  are ignored.
- CI: GitHub Actions runs syntax checks and lightweight smoke tests on Python
  3.10 and 3.11.
- Secrets: working tree and Git history were scanned for common credential
  patterns before publishing.
- Large files: tracked artifacts are small enough for normal GitHub hosting.

## Local Validation

Run these commands before pushing a release branch:

```powershell
.\.venv\Scripts\python.exe smoke_tests.py
py -3.10 -m compileall -q -x "(.venv|tmp|__pycache__|\\.git)" .
git status --short
```

The full reproducibility verifier requires regenerated local experiment outputs
that are intentionally not all committed to GitHub:

```powershell
.\.venv\Scripts\python.exe verify_reproducible_outputs.py
```

## Publishing With GitHub CLI

After authenticating with GitHub CLI:

```powershell
gh auth login
.\scripts\publish_github.ps1
```

The script creates a public repository named `drl-rocket-landing-control`, adds
`origin`, and pushes the current branch. To use a different name:

```powershell
.\scripts\publish_github.ps1 -RepoName your-repo-name
```

If the repository already exists and you want to connect it manually:

```powershell
git remote add origin https://github.com/<owner>/drl-rocket-landing-control.git
git push -u origin main
```

After pushing, enable GitHub security features that are useful for public
research code: dependency graph, Dependabot alerts, and private vulnerability
reporting.
