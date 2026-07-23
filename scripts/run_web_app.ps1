param(
    [int]$Port = 8000,
    [switch]$SkipBrowser
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$AppPython = Join-Path $RepoRoot '.venv-app\Scripts\python.exe'
$FrontendPath = Join-Path $RepoRoot 'frontend'
$FrontendIndex = Join-Path $FrontendPath 'dist\index.html'
$PackageLock = Join-Path $FrontendPath 'package-lock.json'
$NodeModules = Join-Path $FrontendPath 'node_modules'

if (-not (Test-Path -LiteralPath $AppPython)) {
    throw "앱 런타임이 없습니다. 먼저 setup.cmd를 실행해 주세요."
}

if (-not (Test-Path -LiteralPath $NodeModules)) {
    if (-not (Test-Path -LiteralPath $PackageLock)) {
        throw "프론트 패키지 잠금 파일이 없습니다: $PackageLock"
    }
    Push-Location $FrontendPath
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) {
            throw "프론트 패키지 설치에 실패했습니다."
        }
    } finally {
        Pop-Location
    }
}

$BuildRequired = -not (Test-Path -LiteralPath $FrontendIndex)
if (-not $BuildRequired) {
    $BuildTime = (Get-Item -LiteralPath $FrontendIndex).LastWriteTimeUtc
    $LatestSource = Get-ChildItem -LiteralPath (Join-Path $FrontendPath 'src') -Recurse -File |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    $ConfigFiles = @(
        (Join-Path $FrontendPath 'index.html'),
        (Join-Path $FrontendPath 'vite.config.ts'),
        (Join-Path $FrontendPath 'package-lock.json')
    ) | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { Get-Item -LiteralPath $_ }
    $LatestConfig = $ConfigFiles | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    $BuildRequired = (
        ($LatestSource -and $LatestSource.LastWriteTimeUtc -gt $BuildTime) -or
        ($LatestConfig -and $LatestConfig.LastWriteTimeUtc -gt $BuildTime)
    )
}

if ($BuildRequired) {
    Push-Location $FrontendPath
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) {
            throw "프론트 빌드에 실패했습니다."
        }
    } finally {
        Pop-Location
    }
}

$Url = "http://127.0.0.1:$Port"
$BrowserJob = $null
if (-not $SkipBrowser) {
    $BrowserJob = Start-Job -ScriptBlock {
        param($TargetUrl)
        foreach ($Attempt in 1..40) {
            try {
                Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 1 | Out-Null
                Start-Process $TargetUrl
                return
            } catch {
                Start-Sleep -Milliseconds 250
            }
        }
    } -ArgumentList $Url
}

Push-Location $RepoRoot
try {
    & $AppPython -m uvicorn agent.web_server:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
    if ($BrowserJob) {
        Stop-Job -Job $BrowserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $BrowserJob -Force -ErrorAction SilentlyContinue
    }
}
