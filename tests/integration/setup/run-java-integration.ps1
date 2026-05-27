Param(
    [string]$Tag,
    [string]$Sha256,
    [string]$Repo = "causaliq/third-party-binaries"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Tag) {
    throw "Tag is required, e.g. -Tag vcausal-cmd-1.3.0"
}
if (-not $Sha256) {
    throw "Sha256 is required"
}

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")
Set-Location $root

Write-Host "Fetching pinned causal-cmd JAR..."
.\tests\integration\setup\fetch-causal-cmd-jar.ps1 `
    -Repo $Repo `
    -Tag $Tag `
    -ExpectedSha256 $Sha256

$javaDir = Join-Path $root ".artifacts"
$env:CQ_JAVA_DIR = $javaDir
Write-Host "CQ_JAVA_DIR=$javaDir"

Write-Host "Running java integration tests..."
python -m pytest -m java_integration -v --no-cov
