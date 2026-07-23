param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

if (-not $PytestArgs -or $PytestArgs.Count -eq 0) {
    $PytestArgs = @('agent\tests')
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$AppPython = Join-Path $RepoRoot '.venv-app\Scripts\python.exe'
if (Test-Path $AppPython) {
    $Python = $AppPython
} else {
    $Python = 'py'
    $PytestArgs = @('-3.13', '-m', 'pytest') + $PytestArgs
    & $Python @PytestArgs
    exit $LASTEXITCODE
}

& $Python -m pytest @PytestArgs
