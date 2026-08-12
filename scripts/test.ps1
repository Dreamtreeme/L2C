param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

if (-not $PytestArgs -or $PytestArgs.Count -eq 0) {
    $PytestArgs = @('agent\tests', '-q', '-p', 'no:cacheprovider')
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$AppPython = Join-Path $RepoRoot '.venv-app\Scripts\python.exe'
$AppPythonUsable = $false
if (Test-Path -LiteralPath $AppPython) {
    try {
        & $AppPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" 2>$null
        $AppPythonUsable = $LASTEXITCODE -eq 0
    } catch {
        $AppPythonUsable = $false
    }
}
if ($AppPythonUsable) {
    $Python = $AppPython
} else {
    $LauncherUsable = $false
    try {
        & py -3.13 -c "import sys" 2>$null
        $LauncherUsable = $LASTEXITCODE -eq 0
    } catch {
        $LauncherUsable = $false
    }
    if (-not $LauncherUsable) {
        throw 'Python 3.13 실행 환경을 찾을 수 없습니다. setup.cmd -Development를 먼저 실행하세요.'
    }
    & py -3.13 -m pytest @PytestArgs
    exit $LASTEXITCODE
}

& $Python -m pytest @PytestArgs
