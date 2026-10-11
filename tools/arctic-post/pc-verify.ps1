<#
.SYNOPSIS
  One-shot PC verification of the Arctic multithreaded post-processing in ArcticMint (Windows, MSVC + CUDA).

.DESCRIPTION
  NOT RUN IN THE CLOUD: written without access to a Windows machine, an NVIDIA GPU or PowerShell 5.1.
  It was only parse-checked and its pure-logic helpers exercised on PowerShell 7 / Linux. Expect to fix
  small things on the first real run. It stops at the first failure and says what failed and where the
  log is.

  Steps:
    1. Preflight (tools), git ref, ggml submodule, MSVC dev environment.
    2. Configure + build with CMake/Ninja (-DGGML_CUDA=ON, same flags as .github/workflows/release.yml
       on Windows): trellis-cli, post-replay and the CPU-only tests. Run the tests. Then build and run
       the standalone tools/arctic-post tests (post-equiv-test) with MSVC.
    3. Refuse to continue if anything else is using the chosen GPU (default index 1, nvidia-smi order).
    4. trellis-cli --res 1024 (raw dump via TRELLIS_DUMP_POST; -DumpMode Cleaned uses --dump-post) for each phase0 image (CUDA_DEVICE_ORDER=PCI_BUS_ID, --gpu 1).
    5. post-replay on each dump with TRELLIS_POST_HASH=1 at 1 thread, at 24 threads, and with
       TRELLIS_POST_LEGACY=1; compare per-stage hashes; time every run; write summary.md / summary.csv /
       hashes.csv with PASS/FAIL per asset.

  Output goes to <repo>\pcverify-out\<timestamp>\ (override with -OutDir).

.PARAMETER Ref
  Optional git ref to check out first (needs a clean working tree). Default: build whatever is checked out.
.PARAMETER Gpu
  GPU index as nvidia-smi / CUDA_DEVICE_ORDER=PCI_BUS_ID numbers it (default 1 = RTX 4070 SUPER).
.PARAMETER DumpMode
  Raw (default) = TRELLIS_DUMP_POST: the raw decoded mesh (after fill_holes, before weld); the full CLI run
  continues to the GLB, which is slow but makes the replay run the real chain on the real raw mesh.
  Cleaned = trellis-cli --dump-post: stops after the CLI's remesh/clean, so the replay re-remeshes an
  already remeshed mesh (faster, less faithful).
.PARAMETER FrontOnly
  Pass --stop-before-qem to post-replay (skips QEM/UV/GLB; much faster; no GLB comparison).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1
  powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1 -SkipBuild -SkipDump -OutDir D:\pv\last
#>
[CmdletBinding()]
param(
    [string]$Ref = '',
    [string]$RepoRoot = '',
    [string]$BuildDir = '',
    [string]$ModelsDir = 'C:\AI\trellis\models',
    [string]$ImagesDir = 'C:\AI\Expiramental3d\phase0\images',
    [string[]]$Images = @('hard_03_log_cabin', 'hard_02_treasure_chest'),
    [string]$OutDir = '',
    [int]$Gpu = 1,
    [int]$Res = 1024,
    [ValidateRange(2, 1024)]
    [int]$HighThreads = 24,
    [int]$Seed = 1,
    [string]$CudaArch = '86;89',
    [ValidateSet('Cleaned', 'Raw')]
    [string]$DumpMode = 'Raw',
    [int]$BusyMemMiB = 3000,
    [int]$BusyUtilPct = 15,
    [int]$RunTimeoutMin = 90,
    [int]$BuildJobs = 8,
    [switch]$FrontOnly,
    [switch]$SkipBuild,
    [switch]$SkipDump,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$script:StartTime = Get-Date

# ----------------------------------------------------------------------------------------------
# helpers

function Write-Step([string]$msg) {
    Write-Host ''
    Write-Host ('=== ' + $msg) -ForegroundColor Cyan
}

function Write-Info([string]$msg) { Write-Host ('    ' + $msg) }

function Fail([string]$msg) {
    throw ('pc-verify: ' + $msg)
}

function Quote-Arg([string]$a) {
    if ($a -eq '') { return '""' }
    if ($a -match '[\s"]') { return '"' + ($a -replace '"', '\"') + '"' }
    return $a
}

function Join-Args([string[]]$list) {
    $parts = @()
    foreach ($a in $list) { $parts += (Quote-Arg $a) }
    return ($parts -join ' ')
}

# Run a native program with stdout/stderr redirected to files (no PowerShell stderr-to-exception trap),
# an optional temporary environment, a timeout, and peak working-set sampling.
# Returns a PSCustomObject: ExitCode, Seconds, PeakMB, OutFile, ErrFile, TimedOut.
function Invoke-Logged {
    param(
        [string]$Exe,
        [string[]]$ArgList = @(),
        [string]$LogBase,
        [hashtable]$EnvVars = @{},
        [int]$TimeoutMin = 0,
        [string]$WorkDir = ''
    )
    $outFile = $LogBase + '.out.txt'
    $errFile = $LogBase + '.err.txt'
    $saved = @{}
    foreach ($k in $EnvVars.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
        [Environment]::SetEnvironmentVariable($k, [string]$EnvVars[$k], 'Process')
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $peak = 0.0
    $timedOut = $false
    $exit = -1
    try {
        $sp = @{
            FilePath               = $Exe
            RedirectStandardOutput = $outFile
            RedirectStandardError  = $errFile
            NoNewWindow            = $true
            PassThru               = $true
        }
        if ($ArgList.Count -gt 0) { $sp['ArgumentList'] = (Join-Args $ArgList) }
        if ($WorkDir -ne '') { $sp['WorkingDirectory'] = $WorkDir }
        $p = Start-Process @sp
        $procHandle = $p.Handle   # keep the handle so ExitCode is readable after exit
        $lastBeat = 0
        while (-not $p.HasExited) {
            Start-Sleep -Milliseconds 500
            try {
                $p.Refresh()
                $mb = $p.PeakWorkingSet64 / 1MB
                if ($mb -gt $peak) { $peak = $mb }
            } catch { }
            $el = [int]$sw.Elapsed.TotalSeconds
            if ($el -ge $lastBeat + 120) {
                $lastBeat = $el
                Write-Info ('... running {0}s, peak RSS {1:N0} MB' -f $el, $peak)
            }
            if ($TimeoutMin -gt 0 -and $sw.Elapsed.TotalMinutes -gt $TimeoutMin) {
                $timedOut = $true
                try {
                    Start-Process -FilePath 'taskkill.exe' -ArgumentList ('/T /F /PID ' + $p.Id) -NoNewWindow -Wait | Out-Null
                } catch { }
                try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch { }
                break
            }
        }
        $p.WaitForExit()
        if ($timedOut) { $exit = -2 }
        else {
            $p.Refresh()
            if ($null -eq $p.ExitCode) { $exit = -3 } else { $exit = [int]$p.ExitCode }
        }
    } finally {
        $sw.Stop()
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
    }
    return [pscustomobject]@{
        ExitCode = $exit; Seconds = $sw.Elapsed.TotalSeconds; PeakMB = $peak
        OutFile = $outFile; ErrFile = $errFile; TimedOut = $timedOut
    }
}

function Show-Tail([string]$file, [int]$n = 25) {
    if (Test-Path -LiteralPath $file) {
        Write-Host ('--- last lines of ' + $file) -ForegroundColor Yellow
        Get-Content -LiteralPath $file -Tail $n | ForEach-Object { Write-Host ('    ' + $_) }
    }
}

# Run and abort on a non-zero exit.
function Invoke-Checked {
    param([string]$What, [string]$Exe, [string[]]$ArgList, [string]$LogBase, [hashtable]$EnvVars = @{},
          [int]$TimeoutMin = 0, [string]$WorkDir = '')
    Write-Info ($What + ' ...')
    $r = Invoke-Logged -Exe $Exe -ArgList $ArgList -LogBase $LogBase -EnvVars $EnvVars -TimeoutMin $TimeoutMin -WorkDir $WorkDir
    if ($r.ExitCode -ne 0) {
        Show-Tail $r.ErrFile 15
        Show-Tail $r.OutFile 25
        if ($r.TimedOut) { Fail ($What + ' timed out after ' + $TimeoutMin + ' min (log: ' + $LogBase + '.*.txt)') }
        Fail ($What + ' failed with exit code ' + $r.ExitCode + ' (log: ' + $LogBase + '.out.txt / .err.txt)')
    }
    Write-Info ('{0} ok ({1:N1}s)' -f $What, $r.Seconds)
    return $r
}

function Find-Tool([string]$name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    return $null
}

# Import the MSVC x64 developer environment (what ilammy/msvc-dev-cmd does in CI) if cl.exe is not on PATH.
function Ensure-Msvc {
    if ((Find-Tool 'cl.exe') -and ($env:VSCMD_ARG_TGT_ARCH -eq 'x64')) { Write-Info ('x64 MSVC environment already active: ' + (Find-Tool 'cl.exe')); return }
    $pf86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)', 'Process')
    if (-not $pf86) { Fail 'ProgramFiles(x86) is not set; is this Windows?' }
    $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        Fail 'cl.exe is not on PATH and vswhere.exe was not found. Install Visual Studio Build Tools (C++ workload) or run from an "x64 Native Tools Command Prompt" and start powershell from there.'
    }
    $inst = (& $vswhere -latest -products '*' -requires 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' -property installationPath | Select-Object -First 1)
    if (-not $inst) { Fail 'vswhere found no Visual Studio with the MSVC x64 C++ tools component.' }
    $vcvars = Join-Path $inst 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { Fail ('vcvars64.bat not found at ' + $vcvars) }
    Write-Info ('importing MSVC environment from ' + $vcvars)
    $tmpCmd = Join-Path $env:TEMP ('pcverify-vcvars-' + [Guid]::NewGuid().ToString('N') + '.cmd')
    $tmpOut = $tmpCmd + '.out.txt'
    Set-Content -LiteralPath $tmpCmd -Encoding ASCII -Value @('@echo off', ('call "' + $vcvars + '" >nul 2>&1'), 'set')
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/d /c ""' + $tmpCmd + '""') -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
    if ($p.ExitCode -ne 0) { Fail ('vcvars64.bat failed (exit ' + $p.ExitCode + ')') }
    foreach ($line in (Get-Content -LiteralPath $tmpOut)) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
    Remove-Item -LiteralPath $tmpCmd, $tmpOut -Force -ErrorAction SilentlyContinue
    if (-not (Find-Tool 'cl.exe')) { Fail 'cl.exe still not on PATH after importing the MSVC environment.' }
    # VS ships a Ninja; add it if ninja is not otherwise installed.
    if (-not (Find-Tool 'ninja.exe')) {
        $vsNinja = Join-Path $inst 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
        if (Test-Path -LiteralPath (Join-Path $vsNinja 'ninja.exe')) {
            $env:PATH = $vsNinja + ';' + $env:PATH
        }
    }
    Write-Info ('cl.exe: ' + (Find-Tool 'cl.exe'))
}

function Parse-Double([string]$s) {
    return [double]::Parse($s, [System.Globalization.NumberStyles]::Float, $inv)
}

# ---- nvidia-smi ------------------------------------------------------------------------------

function Get-GpuTable {
    $r = Invoke-Logged -Exe 'nvidia-smi.exe' -LogBase (Join-Path $script:Logs 'nvsmi-gpus') -ArgList @(
        '--query-gpu=index,uuid,name,pci.bus_id,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits')
    if ($r.ExitCode -ne 0) { Show-Tail $r.ErrFile; Fail 'nvidia-smi --query-gpu failed' }
    $rows = @()
    foreach ($line in (Get-Content -LiteralPath $r.OutFile)) {
        if (-not $line.Trim()) { continue }
        $f = $line.Split(',') | ForEach-Object { $_.Trim() }
        if ($f.Count -lt 7) { Fail ('unexpected nvidia-smi row: ' + $line) }
        $util = 0; $mem = 0
        if (-not [int]::TryParse($f[6], [ref]$util) -or -not [int]::TryParse($f[4], [ref]$mem)) {
            Fail ('cannot read memory/utilization for GPU ' + $f[0] + ' from nvidia-smi (' + $f[4] + ' / ' + $f[6] + '); cannot prove the GPU is idle')
        }
        $rows += [pscustomobject]@{ Index = [int]$f[0]; Uuid = $f[1]; Name = $f[2]; Bus = $f[3]; MemUsedMiB = $mem; MemTotalMiB = $f[5]; UtilPct = $util }
    }
    return $rows
}

# Refuse when anything other than this script is using GPU $Gpu. WDDM often hides graphics clients
# from nvidia-smi, so besides the compute-apps list this also looks at memory use and utilization.
function Assert-GpuIdle([string]$when) {
    $rows = Get-GpuTable
    $g = $rows | Where-Object { $_.Index -eq $Gpu } | Select-Object -First 1
    if (-not $g) { Fail ('nvidia-smi shows no GPU with index ' + $Gpu + ' (found: ' + (($rows | ForEach-Object { [string]$_.Index }) -join ',') + ')') }
    foreach ($proc in @('trellis-cli', 'trellis-server', 'post-replay')) {
        $running = Get-Process -Name $proc -ErrorAction SilentlyContinue
        if ($running) { Fail ('process "' + $proc + '" is already running (PID ' + (($running | ForEach-Object { $_.Id }) -join ',') + '); refusing to share GPU ' + $Gpu) }
    }
    $r = Invoke-Logged -Exe 'nvidia-smi.exe' -LogBase (Join-Path $script:Logs 'nvsmi-apps') -ArgList @(
        '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits')
    if ($r.ExitCode -eq 0) {
        $mine = @()
        foreach ($line in (Get-Content -LiteralPath $r.OutFile)) {
            if (-not $line.Trim()) { continue }
            $f = $line.Split(',') | ForEach-Object { $_.Trim() }
            if ($f[0] -eq $g.Uuid) { $mine += $line }
        }
        if ($mine.Count -gt 0) { Fail ('GPU ' + $Gpu + ' (' + $g.Name + ') is in use by: ' + ($mine -join ' | ') + ' [' + $when + ']') }
    }
    $maxUtil = $g.UtilPct
    for ($i = 0; $i -lt 3; $i++) {
        Start-Sleep -Seconds 1
        $again = (Get-GpuTable) | Where-Object { $_.Index -eq $Gpu } | Select-Object -First 1
        if ($again.UtilPct -gt $maxUtil) { $maxUtil = $again.UtilPct }
        $g = $again
    }
    if ($g.MemUsedMiB -gt $BusyMemMiB) {
        Fail ('GPU {0} ({1}) already uses {2} MiB (> -BusyMemMiB {3}); something else is on it [{4}]' -f $Gpu, $g.Name, $g.MemUsedMiB, $BusyMemMiB, $when)
    }
    if ($maxUtil -gt $BusyUtilPct) {
        Fail ('GPU {0} ({1}) shows {2}% utilization (> -BusyUtilPct {3}); something else is on it [{4}]' -f $Gpu, $g.Name, $maxUtil, $BusyUtilPct, $when)
    }
    Write-Info ('GPU {0} ({1}, {2}) idle: {3} MiB used, {4}% util [{5}]' -f $Gpu, $g.Name, $g.Bus, $g.MemUsedMiB, $maxUtil, $when)
}

# ---- hashes / logs ---------------------------------------------------------------------------

# Parses "[post-hash] <stage> V=.. F=.. verts=<hex> faces=<hex>" and "[post-hash] bvh tree=<hex>".
# Returns an ordered dictionary stage -> "V=.. F=.. verts=.. faces=.." (the area= column is dropped).
function Parse-StageHashes([string]$file) {
    $d = [ordered]@{}
    foreach ($line in (Get-Content -LiteralPath $file)) {
        if ($line -match '^\s*\[post-hash\]\s+(\S+)\s+(V=\d+\s+F=\d+\s+verts=[0-9a-f]+\s+faces=[0-9a-f]+)') {
            $d[$Matches[1]] = ($Matches[2] -replace '\s+', ' ')
        } elseif ($line -match '^\s*\[post-hash\]\s+(\S+)\s+(tree=[0-9a-f]+)') {
            $d[$Matches[1]] = $Matches[2]
        }
    }
    return $d
}

function Parse-StageTimes([string]$file) {
    $t = @{}
    foreach ($line in (Get-Content -LiteralPath $file)) {
        if ($line -match '^\s*\[(weld|fill|bvh|remesh|clean|clean\+drop|decimate|bake)\s+([0-9.]+)s\]') {
            $t[$Matches[1]] = Parse-Double $Matches[2]
        }
    }
    return $t
}

# SHA-256 of a GLB with the "generated" timestamp (written by mesh_glb.cpp, differs on every run) zeroed.
function Get-GlbHash([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return 'missing' }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $needle = [System.Text.Encoding]::ASCII.GetBytes('"generated":"')
    $lim = [Math]::Min($bytes.Length - $needle.Length, 65536)
    for ($i = 0; $i -lt $lim; $i++) {
        $ok = $true
        for ($j = 0; $j -lt $needle.Length; $j++) { if ($bytes[$i + $j] -ne $needle[$j]) { $ok = $false; break } }
        if ($ok) {
            $k = $i + $needle.Length
            while ($k -lt $bytes.Length -and $bytes[$k] -ne 34) { $bytes[$k] = 48; $k++ }   # 34 = '"', 48 = '0'
            break
        }
    }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $h = $sha.ComputeHash($bytes) } finally { $sha.Dispose() }
    return (($h | ForEach-Object { $_.ToString('x2') }) -join '').Substring(0, 16)
}

# ----------------------------------------------------------------------------------------------
# main

$exitCode = 1
try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { Fail 'this script is for Windows' }
    if ($PSVersionTable.PSVersion.Major -lt 5) { Fail 'PowerShell 5.1 or newer is required' }

    function Full-Path([string]$p) {
        $q = if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path (Get-Location).Path $p }
        $q = [System.IO.Path]::GetFullPath($q)
        if ($q.Length -gt 3) { $q = $q.TrimEnd('\') }   # a trailing backslash would escape the closing quote
        return $q
    }
    if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
    $RepoRoot = Full-Path $RepoRoot
    $ModelsDir = Full-Path $ModelsDir
    $ImagesDir = Full-Path $ImagesDir
    # "-File x.ps1 -Images a,b" arrives as the single string "a,b"
    $Images = @($Images | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'CMakeLists.txt'))) { Fail ('not a repo root: ' + $RepoRoot) }
    if (-not $BuildDir) { $BuildDir = Join-Path $RepoRoot 'build-pcverify' }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    if (-not $OutDir) { $OutDir = Join-Path $RepoRoot ('pcverify-out\' + $stamp) }
    $OutDir = Full-Path $OutDir
    $BuildDir = Full-Path $BuildDir
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $script:Logs = Join-Path $OutDir 'logs'
    New-Item -ItemType Directory -Force -Path $script:Logs | Out-Null
    Write-Host ('pc-verify: output in ' + $OutDir)

    $summary = New-Object System.Collections.ArrayList     # lines for summary.md
    $results = New-Object System.Collections.ArrayList     # per-asset verdict rows
    $csvRows = New-Object System.Collections.ArrayList     # timing rows
    $hashRows = New-Object System.Collections.ArrayList    # stage hash rows
    $buildNotes = New-Object System.Collections.ArrayList

    # ---- 1. preflight ----
    Write-Step '1. preflight'
    foreach ($t in @('git.exe', 'cmake.exe', 'nvidia-smi.exe')) {
        $path = Find-Tool $t
        if (-not $path) { Fail ($t + ' not found on PATH') }
        Write-Info ($t + ': ' + $path)
    }
    if (-not $SkipBuild -or -not $SkipTests) {
        Ensure-Msvc
        if (-not (Find-Tool 'ninja.exe')) { Fail 'ninja.exe not found (install Ninja or Visual Studio CMake tools)' }
    }
    if (-not $SkipBuild) {
        $nvcc = Find-Tool 'nvcc.exe'
        if (-not $nvcc -and $env:CUDA_PATH) {
            $cand = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
            if (Test-Path -LiteralPath $cand) { $env:PATH = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:PATH; $nvcc = $cand }
        }
        if (-not $nvcc) { Fail 'nvcc.exe not found. Install the CUDA toolkit (release.yml uses 13.1 or 12.9.1) and make sure CUDA_PATH or PATH points at it.' }
        Write-Info ('nvcc: ' + $nvcc)
    }
    if ($env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin'))) {
        # the CUDA runtime DLLs must be findable when trellis-cli / post-replay start (also with -SkipBuild)
        $env:PATH = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:PATH
        $x64bin = Join-Path $env:CUDA_PATH 'bin\x64'   # CUDA 13 moved the runtime DLLs here
        if (Test-Path -LiteralPath $x64bin) { $env:PATH = $x64bin + ';' + $env:PATH }
    }
    if (-not (Test-Path -LiteralPath $ModelsDir)) { Fail ('models dir not found: ' + $ModelsDir) }
    if (-not (Test-Path -LiteralPath $ImagesDir)) { Fail ('images dir not found: ' + $ImagesDir) }
    $imageFiles = @{}
    foreach ($name in $Images) {
        $found = $null
        foreach ($ext in @('png', 'jpg', 'jpeg', 'webp')) {
            $c = Join-Path $ImagesDir ($name + '.' + $ext)
            if (Test-Path -LiteralPath $c) { $found = $c; break }
        }
        if (-not $found) { Fail ('image "' + $name + '" (png/jpg/jpeg/webp) not found in ' + $ImagesDir) }
        $imageFiles[$name] = $found
        Write-Info ('image ' + $name + ' -> ' + $found)
    }

    # ---- 2. git ref + submodule ----
    Write-Step '2. git ref and ggml submodule'
    if ($Ref) {
        $dirty = (& git -C $RepoRoot status --porcelain --untracked-files=no)
        if ($dirty) { Fail ('working tree is not clean; commit or stash before using -Ref:' + [Environment]::NewLine + ($dirty -join [Environment]::NewLine)) }
        Invoke-Checked 'git fetch' 'git.exe' @('-C', $RepoRoot, 'fetch', 'origin', $Ref) (Join-Path $script:Logs 'git-fetch') | Out-Null
        Invoke-Checked 'git checkout' 'git.exe' @('-C', $RepoRoot, 'checkout', '--detach', 'FETCH_HEAD') (Join-Path $script:Logs 'git-checkout') | Out-Null
    }
    $head = (& git -C $RepoRoot rev-parse HEAD | Select-Object -First 1)
    $branch = (& git -C $RepoRoot rev-parse --abbrev-ref HEAD | Select-Object -First 1)
    Write-Info ('building ' + $branch + ' @ ' + $head)
    if (-not $SkipBuild) {
        Invoke-Checked 'git submodule update --init --recursive' 'git.exe' @('-C', $RepoRoot, 'submodule', 'update', '--init', '--recursive') (Join-Path $script:Logs 'git-submodule') | Out-Null
        if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'thirdparty\ggml\CMakeLists.txt'))) { Fail 'thirdparty\ggml is still empty after submodule update' }
    }

    # ---- 3. build ----
    $cli = Join-Path $BuildDir 'trellis-cli.exe'
    $replay = Join-Path $BuildDir 'post-replay.exe'
    if (-not $SkipBuild) {
        Write-Step '3. configure and build (MSVC + CUDA, Release, Ninja)'
        $cfgArgs = @('-S', $RepoRoot, '-B', $BuildDir, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                     '-DGGML_CUDA=ON', ('-DCMAKE_CUDA_ARCHITECTURES=' + $CudaArch), '-DGGML_OPENMP=OFF',
                     '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
        Invoke-Checked 'cmake configure' 'cmake.exe' $cfgArgs (Join-Path $script:Logs 'cmake-configure') | Out-Null
        $targets = @('trellis-cli', 'post-replay', 'trellis-test-uv-normal-margin', 'trellis-test-voxel-pbr',
                     'trellis-test-uv-chart-density', 'trellis-test-uv-preset', 'trellis-test-uv-overlap-cover',
                     'trellis-test-quad-topology', 'trellis-test-quad-quality', 'trellis-test-quad-component-cover')
        $bargs = @('--build', $BuildDir, '--target') + $targets + @('--parallel', [string]$BuildJobs)
        Invoke-Checked 'cmake build (CUDA build takes a while)' 'cmake.exe' $bargs (Join-Path $script:Logs 'cmake-build') | Out-Null
        [void]$buildNotes.Add('main build OK: ' + ($targets -join ', '))
    }
    foreach ($exe in @($cli, $replay)) {
        if (-not (Test-Path -LiteralPath $exe)) { Fail ('missing ' + $exe + ' (build it, or drop -SkipBuild)') }
    }

    if (-not $SkipTests) {
        Write-Step '3b. CPU tests (main build) and standalone Arctic tests (MSVC, no CUDA)'
        foreach ($t in @('trellis-test-uv-normal-margin', 'trellis-test-voxel-pbr', 'trellis-test-uv-chart-density',
                         'trellis-test-uv-preset', 'trellis-test-uv-overlap-cover', 'trellis-test-quad-topology',
                         'trellis-test-quad-quality', 'trellis-test-quad-component-cover')) {
            $exe = Join-Path $BuildDir ($t + '.exe')
            if (-not (Test-Path -LiteralPath $exe)) { [void]$buildNotes.Add($t + ': SKIPPED (not built)'); continue }
            $r = Invoke-Logged -Exe $exe -LogBase (Join-Path $script:Logs $t) -TimeoutMin 20
            if ($r.ExitCode -ne 0) { Show-Tail $r.OutFile 20; Show-Tail $r.ErrFile 10; Fail ($t + ' failed, exit ' + $r.ExitCode) }
            [void]$buildNotes.Add($t + ': PASS')
            Write-Info ($t + ': PASS')
        }
        $postDir = Join-Path $RepoRoot 'build-pcverify-post'
        $postSrc = Join-Path $RepoRoot 'tools\arctic-post'
        Invoke-Checked 'configure tools/arctic-post' 'cmake.exe' @('-S', $postSrc, '-B', $postDir, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=cl') (Join-Path $script:Logs 'post-configure') | Out-Null
        Invoke-Checked 'build tools/arctic-post' 'cmake.exe' @('--build', $postDir, '--target', 'post-equiv-test', 'dualgrid-bench', '--parallel', [string]$BuildJobs) (Join-Path $script:Logs 'post-build') | Out-Null
        $r = Invoke-Logged -Exe (Join-Path $postDir 'post-equiv-test.exe') -ArgList @('60') -LogBase (Join-Path $script:Logs 'post-equiv-test') -TimeoutMin 30
        if ($r.ExitCode -ne 0) { Show-Tail $r.ErrFile 20; Fail 'post-equiv-test FAILED: the multithreaded stages differ from the legacy code under MSVC' }
        [void]$buildNotes.Add('post-equiv-test (legacy vs multithreaded, 60 random meshes x 6 thread counts, MSVC): PASS')
        Write-Info 'post-equiv-test: PASS'
    }

    # ---- 4. dumps ----
    Write-Step ('4. trellis-cli dumps (res ' + $Res + ', GPU ' + $Gpu + ', mode ' + $DumpMode + ')')
    $dumps = @{}
    foreach ($name in $Images) {
        $dump = Join-Path $OutDir ($name + '.dump.bin')
        $dumps[$name] = $dump
        if ($SkipDump) {
            if (-not (Test-Path -LiteralPath $dump)) { Fail ('-SkipDump but ' + $dump + ' does not exist (use -OutDir of an earlier run)') }
            Write-Info ('reusing ' + $dump)
            continue
        }
        Assert-GpuIdle ('before dump ' + $name)
        $cliArgs = @('-i', $imageFiles[$name], '-o', (Join-Path $OutDir ($name + '.cli.glb')), '-m', $ModelsDir,
                     '--res', [string]$Res, '--gpu', [string]$Gpu, '--seed', [string]$Seed, '-v')
        $envv = @{ 'CUDA_DEVICE_ORDER' = 'PCI_BUS_ID' }
        if ($DumpMode -eq 'Cleaned') { $cliArgs += @('--dump-post', $dump) }
        else { $envv['TRELLIS_DUMP_POST'] = $dump; $envv['TRELLIS_POST_HASH'] = '1' }
        Invoke-Checked ('trellis-cli ' + $name) $cli $cliArgs (Join-Path $script:Logs ('cli-' + $name)) $envv ($RunTimeoutMin * 2) | Out-Null
        if (-not (Test-Path -LiteralPath $dump)) { Fail ('trellis-cli finished but wrote no dump: ' + $dump) }
        Write-Info ('dump: {0} ({1:N0} MB)' -f $dump, ((Get-Item -LiteralPath $dump).Length / 1MB))
    }

    # ---- 5. replays ----
    Write-Step ('5. post-replay: 1 thread, ' + $HighThreads + ' threads, legacy')
    $configs = @(
        [pscustomobject]@{ Name = 'threads1'; Label = '1 thread'; Env = @{ 'TRELLIS_POST_THREADS' = '1' } },
        [pscustomobject]@{ Name = ('threads' + $HighThreads); Label = ([string]$HighThreads + ' threads'); Env = @{ 'TRELLIS_POST_THREADS' = [string]$HighThreads } },
        [pscustomobject]@{ Name = 'legacy'; Label = 'legacy'; Env = @{ 'TRELLIS_POST_LEGACY' = '1' } }
    )
    $frontStages = @('input', 'weld', 'fill_small_holes', 'bvh', 'remesh', 'clean_mesh', 'drop_components')
    $tailStages = @('decimate_qem', 'weld2', 'fill2', 'drop2')
    $band = [Math]::Max(1, [int]($Res / 512))

    foreach ($name in $Images) {
        $runData = @{}
        foreach ($cfg in $configs) {
            Assert-GpuIdle ('before replay ' + $name + ' ' + $cfg.Name)
            $glb = Join-Path $OutDir ($name + '.' + $cfg.Name + '.glb')
            if (Test-Path -LiteralPath $glb) { Remove-Item -LiteralPath $glb -Force }
            $rargs = @($dumps[$name], $glb, '--geometry-res', [string]$Res, '--band', [string]$band, '--faces', '300000', '--no-audit')
            if ($FrontOnly) { $rargs += '--stop-before-qem' }
            # CUDA_VISIBLE_DEVICES makes the chosen card CUDA device 0 for the (GPU) QEM stage; post-replay
            # never runs ggml, so without this decimate_qem_gpu would land on device 0 = the other card.
            $envv = @{ 'CUDA_DEVICE_ORDER' = 'PCI_BUS_ID'; 'CUDA_VISIBLE_DEVICES' = [string]$Gpu
                       'TRELLIS_POST_HASH' = '1'; 'TRELLIS_POST_TIMING' = '1'; 'TRELLIS_VERBOSE' = '1' }
            foreach ($k in $cfg.Env.Keys) { $envv[$k] = $cfg.Env[$k] }
            Write-Info ($name + ' / ' + $cfg.Label)
            $r = Invoke-Logged -Exe $replay -ArgList $rargs -LogBase (Join-Path $script:Logs ('replay-' + $name + '-' + $cfg.Name)) -EnvVars $envv -TimeoutMin $RunTimeoutMin
            if ($r.ExitCode -ne 0) {
                Show-Tail $r.ErrFile 20; Show-Tail $r.OutFile 30
                Fail ('post-replay ' + $name + ' / ' + $cfg.Label + ' failed (exit ' + $r.ExitCode + $(if ($r.TimedOut) { ', timed out' } else { '' }) + ')')
            }
            $hashes = Parse-StageHashes $r.OutFile
            $times = Parse-StageTimes $r.OutFile
            $allText = (Get-Content -LiteralPath $r.OutFile) + (Get-Content -LiteralPath $r.ErrFile)
            $devLine = ($allText | Where-Object { $_ -match 'decimate_qem_gpu\] current device' } | Select-Object -First 1)
            $fallback = [bool]($allText | Where-Object { $_ -match 'GPU decimation unavailable' } | Select-Object -First 1)
            $glbHash = if ($FrontOnly) { 'n/a' } else { Get-GlbHash $glb }
            $runData[$cfg.Name] = [pscustomobject]@{ Cfg = $cfg; Run = $r; Hashes = $hashes; Times = $times; Glb = $glbHash; Dev = $devLine; Fallback = $fallback }
            Write-Info ('    {0:N1}s, peak {1:N0} MB, {2} stage hashes, glb {3}' -f $r.Seconds, $r.PeakMB, $hashes.Count, $glbHash)
            if ($hashes.Count -eq 0) { Fail ('no [post-hash] lines in ' + $r.OutFile + ' (is this build the Arctic branch?)') }
            $row = [ordered]@{
                asset = $name; config = $cfg.Name; total_s = $r.Seconds.ToString('0.0', $inv); peak_rss_mb = $r.PeakMB.ToString('0', $inv)
                weld_s = ''; fill_s = ''; bvh_s = ''; remesh_s = ''; clean_s = ''; clean_drop_s = ''; decimate_s = ''; bake_s = ''
                glb_sha16_ts_masked = $glbHash; qem_device_log = $devLine; qem_gpu_fallback = $fallback
            }
            $map = @{ 'weld' = 'weld_s'; 'fill' = 'fill_s'; 'bvh' = 'bvh_s'; 'remesh' = 'remesh_s'; 'clean' = 'clean_s'
                      'clean+drop' = 'clean_drop_s'; 'decimate' = 'decimate_s'; 'bake' = 'bake_s' }
            foreach ($k in $map.Keys) { if ($times.ContainsKey($k)) { $row[$map[$k]] = $times[$k].ToString('0.0', $inv) } }
            [void]$csvRows.Add([pscustomobject]$row)
        }

        # compare every config against legacy
        $legacyRun = $runData['legacy']
        $problems = New-Object System.Collections.ArrayList
        $tailProblems = New-Object System.Collections.ArrayList
        foreach ($cfg in $configs) {
            if ($cfg.Name -eq 'legacy') { continue }
            $cur = $runData[$cfg.Name]
            foreach ($st in $frontStages) {
                if (-not $legacyRun.Hashes.Contains($st)) { [void]$problems.Add('legacy run has no hash for stage ' + $st); continue }
                if (-not $cur.Hashes.Contains($st)) { [void]$problems.Add($cfg.Name + ' run has no hash for stage ' + $st); continue }
                if ($legacyRun.Hashes[$st] -ne $cur.Hashes[$st]) { [void]$problems.Add(($cfg.Name + ' differs from legacy at ' + $st + ': ' + $cur.Hashes[$st] + ' vs ' + $legacyRun.Hashes[$st])) }
            }
            if (-not $FrontOnly) {
                foreach ($st in $tailStages) {
                    $haveRef = $legacyRun.Hashes.Contains($st); $haveCur = $cur.Hashes.Contains($st)
                    if ($haveRef -ne $haveCur) { [void]$tailProblems.Add($cfg.Name + ': stage ' + $st + ' present in only one of the runs'); continue }
                    if ($haveRef -and ($legacyRun.Hashes[$st] -ne $cur.Hashes[$st])) { [void]$tailProblems.Add($cfg.Name + ' differs from legacy at ' + $st) }
                }
                if ($legacyRun.Glb -ne $cur.Glb) { [void]$tailProblems.Add($cfg.Name + ' GLB (timestamp masked) ' + $cur.Glb + ' vs legacy ' + $legacyRun.Glb) }
            }
        }
        foreach ($cfg in $configs) {
            foreach ($st in $runData[$cfg.Name].Hashes.Keys) {
                [void]$hashRows.Add([pscustomobject]@{ asset = $name; config = $cfg.Name; stage = $st; hash = $runData[$cfg.Name].Hashes[$st] })
            }
        }
        $verdict = 'PASS'
        if ($problems.Count -gt 0) { $verdict = 'FAIL' }
        elseif ($tailProblems.Count -gt 0) { $verdict = 'FAIL (tail only)' }
        [void]$results.Add([pscustomobject]@{ Asset = $name; Verdict = $verdict; Problems = $problems; TailProblems = $tailProblems; Data = $runData })
        Write-Host ('    ' + $name + ': ' + $verdict) -ForegroundColor $(if ($verdict -eq 'PASS') { 'Green' } else { 'Red' })
    }

    # ---- 6. report ----
    Write-Step '6. report'
    [void]$summary.Add('# Arctic post-MT PC verification')
    [void]$summary.Add('')
    [void]$summary.Add('- date: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' on ' + $env:COMPUTERNAME)
    [void]$summary.Add('- git: `' + $branch + '` @ `' + $head + '`')
    [void]$summary.Add(('- GPU index {0} (CUDA_DEVICE_ORDER=PCI_BUS_ID), res {1}, dump mode {2}{3}' -f $Gpu, $Res, $DumpMode, $(if ($FrontOnly) { ', front chain only' } else { '' })))
    [void]$summary.Add('- stage hashes compared: ' + ($frontStages -join ', ') + $(if ($FrontOnly) { '' } else { '; tail: ' + ($tailStages -join ', ') + ' + GLB sha256 with the generated-timestamp zeroed' }))
    [void]$summary.Add('')
    [void]$summary.Add('## Verdict')
    [void]$summary.Add('')
    [void]$summary.Add('| Asset | Verdict |')
    [void]$summary.Add('|---|---|')
    foreach ($rr in $results) { [void]$summary.Add('| ' + $rr.Asset + ' | **' + $rr.Verdict + '** |') }
    foreach ($rr in $results) {
        if ($rr.Problems.Count -gt 0 -or $rr.TailProblems.Count -gt 0) {
            [void]$summary.Add('')
            [void]$summary.Add('### ' + $rr.Asset + ' problems')
            foreach ($pp in $rr.Problems) { [void]$summary.Add('- FRONT: ' + $pp) }
            foreach ($pp in $rr.TailProblems) { [void]$summary.Add('- TAIL: ' + $pp) }
            if ($rr.Problems.Count -eq 0) {
                [void]$summary.Add('- the front chain (weld..drop_components) is identical; a tail-only difference points at the GPU QEM, xatlas or GLB writer, not the post-MT code. Re-run legacy twice to see whether it is run-to-run noise.')
            }
        }
    }
    [void]$summary.Add('')
    [void]$summary.Add('## Timings (s) and peak RSS')
    [void]$summary.Add('')
    [void]$summary.Add('| Asset | Config | Total | Peak RSS MB | weld | fill | bvh | remesh | clean | clean+drop | decimate | bake | QEM device |')
    [void]$summary.Add('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    foreach ($row in $csvRows) {
        $dev = if ($row.qem_gpu_fallback) { 'CPU FALLBACK' } elseif ($row.qem_device_log) { ($row.qem_device_log -replace '^.*current device ', '') } else { '' }
        [void]$summary.Add(('| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} |' -f $row.asset, $row.config, $row.total_s,
            $row.peak_rss_mb, $row.weld_s, $row.fill_s, $row.bvh_s, $row.remesh_s, $row.clean_s, $row.clean_drop_s, $row.decimate_s, $row.bake_s, $dev))
    }
    [void]$summary.Add('')
    [void]$summary.Add('## Build and tests')
    [void]$summary.Add('')
    foreach ($n in $buildNotes) { [void]$summary.Add('- ' + $n) }
    if ($buildNotes.Count -eq 0) { [void]$summary.Add('- (skipped)') }
    [void]$summary.Add('')
    [void]$summary.Add('Files: `summary.csv` (timings), `hashes.csv` (all stage hashes), `logs\` (every command).')

    Set-Content -LiteralPath (Join-Path $OutDir 'summary.md') -Encoding ASCII -Value $summary
    $csvRows | Export-Csv -LiteralPath (Join-Path $OutDir 'summary.csv') -NoTypeInformation -Encoding ASCII
    $hashRows | Export-Csv -LiteralPath (Join-Path $OutDir 'hashes.csv') -NoTypeInformation -Encoding ASCII
    Write-Host ''
    $summary | ForEach-Object { Write-Host $_ }

    $allPass = $true
    foreach ($rr in $results) { if ($rr.Verdict -ne 'PASS') { $allPass = $false } }
    if ($allPass) { Write-Host ''; Write-Host 'RESULT: PASS (all assets)' -ForegroundColor Green; $exitCode = 0 }
    else { Write-Host ''; Write-Host 'RESULT: FAIL (see above)' -ForegroundColor Red; $exitCode = 1 }
}
catch {
    Write-Host ''
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    $exitCode = 1
}
Write-Host ('elapsed: {0:N1} min' -f ((Get-Date) - $script:StartTime).TotalMinutes)
exit $exitCode
