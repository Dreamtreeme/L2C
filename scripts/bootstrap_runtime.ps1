param(
    [string]$PythonPath = "",
    [switch]$SkipBrowserInstall,
    [switch]$SkipAssetDownload,
    [switch]$NoInstallPython,
    [switch]$Development,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$MinimumFreeBytes = 12GB
$MinimumVramMiB = 8192
$MinimumCuda13DriverMajor = 580
$PinnedPythonVersion = '3.13.14'
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PinnedPythonVersion/python-$PinnedPythonVersion-amd64.exe"
$PythonInstallerSha256 = 'c54d9b9bbb8a36e6489363ddd01139707fd781d72f1f9e90c7ec65d0061368e0'

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

function Test-Python313 {
    param([string]$Candidate)

    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate)) {
        return $false
    }
    try {
        $Version = (& $Candidate -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null)
    } catch {
        return $false
    }
    return $LASTEXITCODE -eq 0 -and $Version.Trim().StartsWith('3.13.')
}

function Find-Python313 {
    $Candidates = @()
    if ($PythonPath) {
        $Candidates += $PythonPath
    }
    $Candidates += Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        try {
            $Discovered = (& $PyLauncher.Source -3.13 -c "import sys; print(sys.executable)" 2>$null)
        } catch {
            $Discovered = ""
        }
        if ($LASTEXITCODE -eq 0 -and $Discovered) {
            $Candidates += $Discovered.Trim()
        }
    }

    foreach ($Candidate in $Candidates | Select-Object -Unique) {
        if (Test-Python313 $Candidate) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    return ""
}

function Find-NvidiaSmi {
    $Command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    $DefaultPath = 'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    if (Test-Path -LiteralPath $DefaultPath) {
        return $DefaultPath
    }
    return ""
}

function Install-Python313 {
    $InstallerPath = Join-Path $env:TEMP "python-$PinnedPythonVersion-amd64.exe"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $PythonInstallerUrl -OutFile $InstallerPath
        $ActualHash = (Get-FileHash -LiteralPath $InstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $PythonInstallerSha256) {
            throw "Python 설치 파일의 SHA-256이 공식 값과 다릅니다: $ActualHash"
        }

        $Process = Start-Process `
            -FilePath $InstallerPath `
            -ArgumentList @(
                '/quiet',
                'InstallAllUsers=0',
                'PrependPath=1',
                'Include_launcher=1',
                'Include_test=0'
            ) `
            -Wait `
            -PassThru `
            -WindowStyle Hidden
        if ($Process.ExitCode -notin @(0, 3010)) {
            throw "Python 설치 프로그램이 실패했습니다: exit_code=$($Process.ExitCode)"
        }
    } finally {
        if (Test-Path -LiteralPath $InstallerPath) {
            Remove-Item -LiteralPath $InstallerPath -Force
        }
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "L2C는 64비트 Windows가 필요합니다."
}

$DriveRoot = [System.IO.Path]::GetPathRoot($RepoRoot)
$Drive = [System.IO.DriveInfo]::new($DriveRoot)
$FreeGiB = [math]::Round($Drive.AvailableFreeSpace / 1GB, 1)
if ($Drive.AvailableFreeSpace -lt $MinimumFreeBytes) {
    throw "설치 드라이브에 최소 12GB가 필요합니다. 현재 여유 공간: ${FreeGiB}GB"
}
Write-Host "디스크 점검 완료: ${FreeGiB}GB 사용 가능"

$NvidiaSmi = Find-NvidiaSmi
if (-not $NvidiaSmi) {
    throw "NVIDIA 드라이버를 찾을 수 없습니다. 드라이버 설치 후 setup.cmd를 다시 실행하세요."
}
$GpuInfo = (
    & $NvidiaSmi `
        --query-gpu=name,driver_version,memory.total `
        --format=csv,noheader,nounits 2>$null
)
if ($LASTEXITCODE -ne 0 -or -not $GpuInfo) {
    throw "NVIDIA GPU 정보를 읽지 못했습니다: $NvidiaSmi"
}
$SupportedGpus = @()
foreach ($GpuRow in $GpuInfo) {
    $Parts = @($GpuRow -split ',' | ForEach-Object { $_.Trim() })
    if ($Parts.Count -lt 3) {
        continue
    }
    $DriverMajor = 0
    $VramMiB = 0
    [void][int]::TryParse(($Parts[1] -split '\.')[0], [ref]$DriverMajor)
    [void][int]::TryParse($Parts[2], [ref]$VramMiB)
    if (
        $DriverMajor -ge $MinimumCuda13DriverMajor -and
        $VramMiB -ge $MinimumVramMiB
    ) {
        $SupportedGpus += @{
            Name = $Parts[0]
            Driver = $Parts[1]
            VramMiB = $VramMiB
        }
    }
}
if (-not $SupportedGpus) {
    throw (
        "CUDA 13 실행에는 NVIDIA 드라이버 580 이상과 VRAM 8GB 이상인 GPU가 필요합니다. " +
        "감지 결과: $($GpuInfo -join '; ')"
    )
}
$SupportedGpus | ForEach-Object {
    $VramGiB = [math]::Round($_.VramMiB / 1024, 1)
    Write-Host "GPU 점검 완료: $($_.Name), driver=$($_.Driver), VRAM=${VramGiB}GB"
}

$ResolvedPython = Find-Python313
if (-not $ResolvedPython) {
    if ($NoInstallPython) {
        throw "Python 3.13을 찾을 수 없습니다."
    }
    if ($DryRun) {
        Write-Host "Python $PinnedPythonVersion 설치 파일을 python.org에서 받아 설치할 예정입니다."
    } else {
        Install-Python313
        $ResolvedPython = Find-Python313
        if (-not $ResolvedPython) {
            throw "설치 후에도 Python $PinnedPythonVersion을 찾을 수 없습니다."
        }
    }
}

if ($ResolvedPython) {
    Write-Host "Python 점검 완료: $ResolvedPython"
}

$EnvPath = Join-Path $RepoRoot '.env'
if (-not (Test-Path -LiteralPath $EnvPath)) {
    if ($DryRun) {
        Write-Host ".env.example을 .env로 복사할 예정입니다."
    } else {
        Copy-Item -LiteralPath (Join-Path $RepoRoot '.env.example') -Destination $EnvPath
        Write-Host ".env 파일을 생성했습니다."
    }
}

$SetupParameters = @{
    PythonPath = $ResolvedPython
}
if ($SkipBrowserInstall) {
    $SetupParameters.SkipBrowserInstall = $true
}
if ($SkipAssetDownload) {
    $SetupParameters.SkipAssetDownload = $true
}
if ($Development) {
    $SetupParameters.Development = $true
}

if ($DryRun) {
    $DisplayPython = $ResolvedPython
    if (-not $DisplayPython) {
        $DisplayPython = '<Python 3.13.14 자동 설치 경로>'
    }
    $DisplayArguments = @('-PythonPath', $DisplayPython)
    if ($SkipBrowserInstall) {
        $DisplayArguments += '-SkipBrowserInstall'
    }
    if ($SkipAssetDownload) {
        $DisplayArguments += '-SkipAssetDownload'
    }
    if ($Development) {
        $DisplayArguments += '-Development'
    }
    Write-Host "실행 예정: scripts\setup_runtime.ps1 $($DisplayArguments -join ' ')"
    exit 0
}

& (Join-Path $PSScriptRoot 'setup_runtime.ps1') @SetupParameters
if ($LASTEXITCODE -ne 0) {
    throw "L2C 런타임 설치에 실패했습니다."
}

Write-Host ""
Write-Host "L2C 설치가 완료됐습니다."
Write-Host "수집 실행 전 .env의 GEMINI_API_KEY를 설정하세요."
Write-Host "실행: .\run.cmd"
