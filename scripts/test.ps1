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
if (-not (Test-Path -LiteralPath $AppPython)) {
    throw '앱 Python 환경을 찾을 수 없습니다. setup.cmd를 먼저 실행하세요.'
}

& $AppPython -m pytest @PytestArgs
