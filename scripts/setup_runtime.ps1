param(
    [string]$PythonPath = "",
    [switch]$SkipBrowserInstall,
    [switch]$SkipAssetDownload
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Invoke-CheckedCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "명령 실행에 실패했습니다: $FilePath $($ArgumentList -join ' ')"
    }
}

if (-not $PythonPath) {
    $DefaultPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
    if (Test-Path -LiteralPath $DefaultPython) {
        $PythonPath = $DefaultPython
    } else {
        $PythonPath = (& py -3.13 -c "import sys; print(sys.executable)").Trim()
    }
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python 3.13 실행 파일을 찾을 수 없습니다: $PythonPath"
}

$PythonVersion = (& $PythonPath -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if (-not $PythonVersion.StartsWith('3.13.')) {
    throw "Python 3.13이 필요합니다. 현재 버전: $PythonVersion"
}

function Install-Environment {
    param(
        [string]$EnvironmentPath,
        [string]$RequirementsPath
    )

    Invoke-CheckedCommand $PythonPath @('-m', 'venv', $EnvironmentPath)
    $EnvironmentPython = Join-Path $EnvironmentPath 'Scripts\python.exe'
    Invoke-CheckedCommand $EnvironmentPython @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-CheckedCommand $EnvironmentPython @('-m', 'pip', 'install', '-r', $RequirementsPath)
    Invoke-CheckedCommand $EnvironmentPython @('-m', 'pip', 'check')
}

$AppEnvironment = Join-Path $RepoRoot '.venv-app'
$OcrEnvironment = Join-Path $RepoRoot '.venv-ocr'

Install-Environment $AppEnvironment (Join-Path $RepoRoot 'requirements-dev.txt')
Install-Environment $OcrEnvironment (Join-Path $RepoRoot 'requirements-ocr.txt')

$AppPython = Join-Path $AppEnvironment 'Scripts\python.exe'
$OcrPython = Join-Path $OcrEnvironment 'Scripts\python.exe'

if (-not $SkipBrowserInstall) {
    Invoke-CheckedCommand $AppPython @('-m', 'playwright', 'install', 'chromium')
}

$CompatibilityScript = Join-Path $RepoRoot 'scripts\check_runtime_compat.py'
Invoke-CheckedCommand $AppPython @($CompatibilityScript, '--profile', 'app')
Invoke-CheckedCommand $OcrPython @($CompatibilityScript, '--profile', 'ocr')

if (-not $SkipAssetDownload) {
    $AssetScript = Join-Path $RepoRoot 'scripts\prepare_runtime_assets.py'
    Invoke-CheckedCommand $AppPython @($AssetScript, '--component', 'omniparser')
    Invoke-CheckedCommand $OcrPython @($AssetScript, '--component', 'paddleocr')
}

Write-Host "Python $PythonVersion 런타임 구성이 완료됐습니다."
Write-Host "앱: $AppPython"
Write-Host "OCR: $OcrPython"
