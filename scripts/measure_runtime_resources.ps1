param(
    [string]$OutputPath = "",
    [string]$Command = "",
    [string[]]$CommandArguments = @(),
    [int]$SampleIntervalMs = 250
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_utf8.ps1"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputPath) {
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputPath = Join-Path $RepoRoot "logs\runtime_resources_$Stamp.json"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)

if (-not ('L2CMemoryStatus' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class L2CMemoryStatus
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public class MemoryStatusEx
    {
        public uint Length = (uint)Marshal.SizeOf(typeof(MemoryStatusEx));
        public uint MemoryLoad;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern bool GlobalMemoryStatusEx(
        [In, Out] MemoryStatusEx buffer
    );

    public static long UsedPhysicalBytes()
    {
        var status = new MemoryStatusEx();
        if (!GlobalMemoryStatusEx(status))
        {
            throw new System.ComponentModel.Win32Exception(
                Marshal.GetLastWin32Error()
            );
        }
        return checked((long)(status.TotalPhysical - status.AvailablePhysical));
    }
}
'@
}

function Get-DirectorySizeBytes {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    try {
        return [int64](
            Get-ChildItem -LiteralPath $Path -Recurse -File -Force |
                Measure-Object -Property Length -Sum
        ).Sum
    } catch {
        Write-Warning "디렉터리 용량을 읽지 못했습니다: $Path"
        return 0
    }
}

function Find-NvidiaSmi {
    $CommandInfo = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($CommandInfo) {
        return $CommandInfo.Source
    }
    $DefaultPath = 'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    if (Test-Path -LiteralPath $DefaultPath) {
        return $DefaultPath
    }
    return ""
}

function Get-SystemUsedMemoryBytes {
    return [int64][L2CMemoryStatus]::UsedPhysicalBytes()
}

function Get-GpuSnapshot {
    param([string]$NvidiaSmi)

    if (-not $NvidiaSmi) {
        return @()
    }
    $Rows = (
        & $NvidiaSmi `
            --query-gpu=name,driver_version,memory.total,memory.used `
            --format=csv,noheader,nounits 2>$null
    )
    if ($LASTEXITCODE -ne 0) {
        return @()
    }
    return @(
        foreach ($Row in $Rows) {
            $Parts = @($Row -split ',' | ForEach-Object { $_.Trim() })
            if ($Parts.Count -lt 4) {
                continue
            }
            @{
                name = $Parts[0]
                driver_version = $Parts[1]
                memory_total_mib = [int]$Parts[2]
                memory_used_mib = [int]$Parts[3]
            }
        }
    )
}

$NvidiaSmi = Find-NvidiaSmi
$InitialGpu = @(Get-GpuSnapshot $NvidiaSmi)
$InitialSystemMemory = Get-SystemUsedMemoryBytes
$PeakSystemMemory = $InitialSystemMemory
$PeakGpuUsedMiB = @($InitialGpu | ForEach-Object { $_.memory_used_mib })
$StartedAt = Get-Date
$ExitCode = $null

if ($Command) {
    $Process = Start-Process `
        -FilePath $Command `
        -ArgumentList $CommandArguments `
        -WorkingDirectory $RepoRoot `
        -PassThru `
        -WindowStyle Hidden
    while (-not $Process.HasExited) {
        Start-Sleep -Milliseconds ([math]::Max(100, $SampleIntervalMs))
        $PeakSystemMemory = [math]::Max(
            $PeakSystemMemory,
            (Get-SystemUsedMemoryBytes)
        )
        $Gpu = @(Get-GpuSnapshot $NvidiaSmi)
        for ($Index = 0; $Index -lt $Gpu.Count; $Index++) {
            if ($Index -ge $PeakGpuUsedMiB.Count) {
                $PeakGpuUsedMiB += 0
            }
            $PeakGpuUsedMiB[$Index] = [math]::Max(
                $PeakGpuUsedMiB[$Index],
                [int]$Gpu[$Index].memory_used_mib
            )
        }
        $Process.Refresh()
    }
    $ExitCode = $Process.ExitCode
}

$FinishedAt = Get-Date
$EndGpu = @(Get-GpuSnapshot $NvidiaSmi)
$RuntimeSizes = @{
    app_environment = Get-DirectorySizeBytes (Join-Path $RepoRoot '.venv-app')
    ocr_environment = Get-DirectorySizeBytes (Join-Path $RepoRoot '.venv-ocr')
    models = Get-DirectorySizeBytes (Join-Path $RepoRoot 'models')
    playwright_browsers = Get-DirectorySizeBytes (
        Join-Path $env:LOCALAPPDATA 'ms-playwright'
    )
}
$InstallBytes = [int64](($RuntimeSizes.Values | Measure-Object -Sum).Sum)
$ProjectDataBytes = Get-DirectorySizeBytes (Join-Path $RepoRoot 'data')

$Report = @{
    schema_version = 1
    measured_at = $FinishedAt.ToString('o')
    command = $Command
    command_arguments = $CommandArguments
    exit_code = $ExitCode
    elapsed_sec = [math]::Round(
        ($FinishedAt - $StartedAt).TotalSeconds,
        3
    )
    install = @{
        total_bytes = $InstallBytes
        total_gib = [math]::Round($InstallBytes / 1GB, 3)
        components_bytes = $RuntimeSizes
        project_data_bytes = $ProjectDataBytes
        project_data_gib = [math]::Round($ProjectDataBytes / 1GB, 3)
    }
    memory = @{
        baseline_system_used_bytes = $InitialSystemMemory
        peak_system_used_bytes = $PeakSystemMemory
        peak_system_delta_gib = [math]::Round(
            ($PeakSystemMemory - $InitialSystemMemory) / 1GB,
            3
        )
    }
    gpu = @{
        initial = $InitialGpu
        final = $EndGpu
        peak_memory_used_mib = $PeakGpuUsedMiB
        peak_memory_delta_mib = @(
            for ($Index = 0; $Index -lt $PeakGpuUsedMiB.Count; $Index++) {
                $Baseline = if ($Index -lt $InitialGpu.Count) {
                    [int]$InitialGpu[$Index].memory_used_mib
                } else {
                    0
                }
                [math]::Max(0, $PeakGpuUsedMiB[$Index] - $Baseline)
            }
        )
    }
}

$OutputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$Report | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "RESOURCE_REPORT=$OutputPath"
if ($null -ne $ExitCode -and $ExitCode -ne 0) {
    exit $ExitCode
}
