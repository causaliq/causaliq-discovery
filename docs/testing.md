# Testing

> **Audience: contributors only.**  This document describes how to run
> and extend the test suite for causaliq-discovery.  If you are an
> end-user looking for information on installing and configuring
> supported algorithms, see the
> [User Guide](userguide/algorithms.md).

## Test categories

| Category | Location | Runs in CI | Coverage |
|---|---|---|---|
| Unit | `tests/unit/` | Always | Required (100%) |
| Functional | `tests/functional/` | Always | Required (100%) |
| Integration | `tests/integration/` | Always | Required (100%) |
| R integration | `tests/integration/bnlearn/` | Manual only | Excluded |
| Java integration | `tests/integration/tetrad/` | Manual only | Excluded |

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

### Java integration tests

Java integration tests run real Tetrad FGES calls via the pinned
`causal-cmd` JAR. They are skipped automatically when Java is not
available or `CQ_JAVA_DIR` is not configured.

Fetch the JAR from the private third-party binaries release:

The release asset is currently named
`causal-cmd-1.3.0-jar-with-dependencies.jar`.
The helper script stores it locally as
`./.artifacts/causal-cmd-1.3.0.jar` for adapter compatibility.

```powershell
.\tests\integration\setup\fetch-causal-cmd-jar.ps1 `
	-Tag <release-tag> `
	-ExpectedSha256 <sha256>
```

Set `CQ_JAVA_DIR` to the downloaded JAR directory, then run:

```powershell
.\scripts\activate.ps1; python -m pytest -m java_integration -v --no-cov
```

For an end-to-end local run, use:

```powershell
.\tests\integration\setup\run-java-integration.ps1 `
	-Tag <release-tag> `
	-Sha256 <sha256>
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

## Java integration tests on GitHub Actions

Java integration tests do not run automatically on every commit. They are
triggered manually when needed (e.g. before a release, or after changes
to the Tetrad adapter):

1. Go to the repository on GitHub.
2. Click the **Actions** tab.
3. Select **Java Integration** in the left sidebar.
4. Click **Run workflow**.

By default, `java_profile=tetrad_fges_1_3_0` is selected and the
workflow uses pinned values for tag, asset name and checksum.
This supports one-click runs with no manual input.

To run another version, either:

- select `java_profile=custom` and provide all three override inputs
   (`jar_tag`, `jar_asset_name`, `jar_sha256`), or
- keep the default profile and fill one or more override inputs.

### Prerequisite: `THIRD_PARTY_BINARIES_TOKEN` repository secret

The workflow downloads the pinned JAR from the private
`causaliq/third-party-binaries` repository.  A repository secret
must be configured before the workflow can run.

**Step 1 — Create a fine-grained personal access token (PAT)**

1. Go to **GitHub → Settings → Developer settings →
   Personal access tokens → Fine-grained tokens**.
2. Click **Generate new token**.
3. Set:
   - **Token name**: `causaliq-third-party-binaries-read`
   - **Description**: Read-only token for downloading pinned Java
     artefacts from `causaliq/third-party-binaries` for
     causaliq-discovery integration tests.
   - **Resource owner**: `causaliq`
   - **Repository access**: only `causaliq/third-party-binaries`
   - **Permissions**: Contents → Read-only; Metadata → Read-only
     (required automatically)
4. Click **Generate token** and copy the value immediately
   (it is shown only once).

**Step 2 — Add the token as a repository secret**

1. Go to the **causaliq-discovery** repository on GitHub.
2. Click **Settings → Secrets and variables → Actions**.
3. Click **New repository secret**.
4. Name: `THIRD_PARTY_BINARIES_TOKEN`; Value: paste the copied token.
5. Click **Add secret**.

The workflow runs on Ubuntu and Windows with Python 3.11 and Java 17.
