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
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = 'python'
}

& $Python -m pytest @PytestArgs