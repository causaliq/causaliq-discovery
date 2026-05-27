Param(
    [string]$Repo = "causaliq/third-party-binaries",
    [string]$Tag = "",
    [string]$AssetName = "causal-cmd-1.3.0-jar-with-dependencies.jar",
    [string]$OutPath = "./.artifacts/causal-cmd-1.3.0.jar",
    [string]$Token = "",
    [string]$ExpectedSha256 = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Token) {
    $Token = $env:GH_TOKEN
}
if (-not $Token) {
    $Token = $env:GITHUB_TOKEN
}
if (-not $ExpectedSha256) {
    $ExpectedSha256 = $env:CQ_JAVA_SHA256
}

$resolvedOutPath = Resolve-Path -LiteralPath "." | ForEach-Object {
    Join-Path $_ $OutPath
}
$outDir = Split-Path -Parent $resolvedOutPath
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$directUrl = $env:CQ_JAVA_ARTIFACT_URL
if ($directUrl) {
    Write-Host "Downloading causal-cmd JAR from CQ_JAVA_ARTIFACT_URL..."
    $headers = @{}
    if ($Token) {
        $headers["Authorization"] = "Bearer $Token"
    }
    Invoke-WebRequest -Uri $directUrl -OutFile $resolvedOutPath -Headers $headers
} else {
    if (-not $Tag) {
        throw "Tag is required when CQ_JAVA_ARTIFACT_URL is not set."
    }
    if (-not $Token) {
        throw "Set GH_TOKEN or GITHUB_TOKEN for private release access."
    }

    Write-Host "Resolving release '$Tag' in $Repo..."
    $apiBase = "https://api.github.com/repos/$Repo/releases/tags/$Tag"
    $headers = @{
        "Authorization" = "Bearer $Token"
        "Accept" = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
    $release = Invoke-RestMethod -Uri $apiBase -Headers $headers
    $asset = $release.assets | Where-Object { $_.name -eq $AssetName } |
        Select-Object -First 1
    if (-not $asset) {
        throw "Asset '$AssetName' not found in release '$Tag'."
    }

    Write-Host "Downloading asset '$AssetName'..."
    $dlHeaders = @{
        "Authorization" = "Bearer $Token"
        "Accept" = "application/octet-stream"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
    Invoke-WebRequest -Uri $asset.url -OutFile $resolvedOutPath -Headers $dlHeaders
}

if ($ExpectedSha256) {
    $actual = (Get-FileHash -LiteralPath $resolvedOutPath -Algorithm SHA256).Hash
    if ($actual -ne $ExpectedSha256) {
        throw (
            "SHA256 mismatch for $resolvedOutPath. " +
            "Expected $ExpectedSha256, got $actual"
        )
    }
    Write-Host "SHA256 verified."
}

Write-Host "Downloaded causal-cmd JAR: $resolvedOutPath"
Write-Host "Set CQ_JAVA_DIR=$outDir before running tests."
