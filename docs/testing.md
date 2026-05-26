# Testing

## Test categories

| Category | Location | Runs in CI | Coverage |
|---|---|---|---|
| Unit | `tests/unit/` | Always | Required (100%) |
| Functional | `tests/functional/` | Always | Required (100%) |
| Integration | `tests/integration/` | Always | Required (100%) |
| R integration | `tests/integration/bnlearn/` | Manual only | Excluded |

## Running tests locally

### Unit, functional and integration tests

```powershell
.\scripts\activate.ps1; python -m pytest tests/ -v
```

### R integration tests

R integration tests call real R/bnlearn via subprocess and are skipped
automatically when R or bnlearn is not installed.

```powershell
.\scripts\activate.ps1; python -m pytest -m r_integration -v --no-cov
```

### All CI checks (formatting, types, coverage)

```powershell
.\scripts\activate.ps1; .\scripts\check_ci.ps1
```

## R integration tests on GitHub Actions

R integration tests do not run automatically on every commit. They are
triggered manually when needed (e.g. before a release, or after changes
to the bnlearn adapter):

1. Go to the repository on GitHub.
2. Click the **Actions** tab.
3. Select **R Integration** in the left sidebar.
4. Click **Run workflow** → **Run workflow**.

The workflow runs on Ubuntu and Windows with Python 3.11. R packages are
cached between runs to avoid reinstalling bnlearn from scratch each time.
