# PardaDB installer for Windows (PowerShell 5.1 and PowerShell 7).
#
# One command (default options):
#   irm https://raw.githubusercontent.com/deadhearth01/pardadb/main/FULL_CODE/scripts/install.ps1 | iex
# With options (iex cannot pass parameters, so build a script block instead):
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/deadhearth01/pardadb/main/FULL_CODE/scripts/install.ps1))) -Data C:\path\to\B1_Smart_Data_Masking_PII -WithAI
# From a clone (installs in place, does not pull):
#   powershell -ExecutionPolicy Bypass -File .\FULL_CODE\scripts\install.ps1
#
# Safe to re-run: it updates the checkout with a fast-forward pull, re-syncs the
# locked dependencies and never overwrites an existing FULL_CODE\.env.
# It never asks for administrator rights; when a tool is missing it prints the
# winget command to run.
#
# Everything runs inside Install-PardaDB. With "irm | iex" the script runs in the
# caller's own session, so we must not change their $ErrorActionPreference or
# call "exit" (that would close their PowerShell window): errors are thrown.

param(
    [string]$Dir = $(if ($env:PARDADB_INSTALL_DIR) { $env:PARDADB_INSTALL_DIR } else { Join-Path $HOME 'pardadb' }),
    [string]$Branch = $(if ($env:PARDADB_INSTALL_BRANCH) { $env:PARDADB_INSTALL_BRANCH } else { 'main' }),
    [string]$Data = $env:PARDADB_INSTALL_DATA,
    [string]$Source = $env:PARDADB_INSTALL_SOURCE,
    [switch]$WithAI = ($env:PARDADB_INSTALL_WITH_AI -eq '1'),
    [switch]$StartDatabases = ($env:PARDADB_INSTALL_START_DATABASES -eq '1'),
    [switch]$NoUv = ($env:PARDADB_INSTALL_NO_UV -eq '1')
)

$script:DefaultSource = 'https://github.com/deadhearth01/pardadb.git'
$script:UvInstallerUrl = 'https://astral.sh/uv/install.ps1'
$script:TorchCpuIndex = 'https://download.pytorch.org/whl/cpu'
$script:TorchCudaIndex = 'https://download.pytorch.org/whl/cu128'
$script:DashboardUrl = 'http://127.0.0.1:8765'

# Write-Host on purpose: these are coloured messages for a person at a console,
# not pipeline output (with "irm | iex" anything returned would be printed twice).
function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host " ok $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Host "warning: $Message" -ForegroundColor Yellow }

# $IsWindows only exists in PowerShell 6+; Windows PowerShell 5.1 is always Windows.
function Test-IsWindows {
    if ($PSVersionTable.PSEdition -eq 'Desktop') { return $true }
    return [bool]$IsWindows
}

# Runs a native program and throws if it fails. PowerShell does not stop on a
# non-zero exit code by itself, and PowerShell 5.1 can turn harmless stderr
# output (git progress) into errors, so the preference is relaxed around the call.
function Invoke-Native {
    param([string]$What, [scriptblock]$Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($exitCode -ne 0) { throw "$What failed (exit code $exitCode). See the output above." }
}

# Runs a native program quietly and returns $true if it exited with 0. Used for
# probes (is there a GPU, does a branch exist) where failure is a normal answer.
function Test-Native([scriptblock]$Command) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

# Finds a Python 3.10+ and returns the command as an array (exe plus arguments).
# The "python" found on a fresh Windows can be the Microsoft Store stub, which
# exists but does not run, so every candidate is actually executed.
function Find-Python {
    $candidates = @(@('py', '-3'), @('python'), @('python3'))
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $exe = $candidate[0]
        $pyArgs = @($candidate | Select-Object -Skip 1)
        if (Test-Native { & $exe @pyArgs -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' }) {
            return , $candidate
        }
    }
    return $null
}

# uv installs into %USERPROFILE%\.local\bin (older releases: %USERPROFILE%\.cargo\bin),
# which a running session may not have on PATH yet.
function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $name = 'uv'
    if (Test-IsWindows) { $name = 'uv.exe' }
    foreach ($folder in @((Join-Path $HOME '.local\bin'), (Join-Path $HOME '.cargo\bin'))) {
        $path = Join-Path $folder $name
        if (Test-Path -LiteralPath $path) { return $path }
    }
    return $null
}

# Installs uv with Astral's official installer, the documented way to get uv.
# It runs in a child PowerShell so its own settings cannot leak into ours. Its
# messages go to Out-Host: anything left in the output stream would become part
# of this function's return value, and the caller would try to run the text as uv.
function Install-Uv {
    Write-Step "Installing uv (Python package manager) from $script:UvInstallerUrl"
    $shell = 'powershell'
    if ($PSVersionTable.PSEdition -eq 'Core') { $shell = 'pwsh' }
    try {
        Invoke-Native 'The uv installer' {
            & $shell -NoProfile -ExecutionPolicy Bypass -Command "irm $script:UvInstallerUrl | iex" | Out-Host
        }
    } catch {
        return $null
    }
    return Find-Uv
}

# Clones the repo, or fast-forwards an existing clone. A fast-forward pull never
# rewrites local work: if the user changed tracked files, git refuses and we stop.
function Update-Checkout([string]$From, [string]$Name, [string]$Into) {
    if (Test-Path -LiteralPath (Join-Path $Into '.git')) {
        Write-Step "Updating the existing checkout in $Into (branch $Name)"
        Invoke-Native "git fetch" { git -C $Into fetch --quiet origin $Name }
        if (Test-Native { git -C $Into rev-parse --verify --quiet "refs/heads/$Name" }) {
            Invoke-Native "git checkout $Name" { git -C $Into checkout --quiet $Name }
        } else {
            Invoke-Native "git checkout $Name" { git -C $Into checkout --quiet -b $Name --track "origin/$Name" }
        }
        try {
            Invoke-Native "git pull --ff-only" { git -C $Into pull --ff-only --quiet origin $Name }
        } catch {
            throw "Could not fast-forward $Into. Commit or discard your local changes there, or choose another -Dir."
        }
    } elseif ((Test-Path -LiteralPath $Into) -and (Get-ChildItem -LiteralPath $Into -Force | Select-Object -First 1)) {
        throw "$Into exists, is not empty and is not a git checkout. Choose another -Dir or remove it."
    } else {
        Write-Step "Cloning $From (branch $Name) into $Into"
        $parent = Split-Path -Parent $Into
        if ($parent -and -not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
        Invoke-Native "git clone" { git clone --quiet --branch $Name $From $Into }
    }
    $head = git -C $Into log -1 --format='%h %s'
    Write-Ok "Source ready: $head"
}

# Path of an executable inside a venv (Scripts\x.exe on Windows, bin/x elsewhere).
function Get-VenvExe([string]$Venv, [string]$Name) {
    if (Test-IsWindows) { return Join-Path $Venv "Scripts\$Name.exe" }
    return Join-Path $Venv "bin/$Name"
}

# Links the problem pack into FULL_CODE\data, where PardaDB looks by default. A
# directory junction needs no administrator rights or Developer Mode, unlike a
# symbolic link. A real folder already there is left alone.
function Set-DataLink([string]$Code, [string]$Pack) {
    if (-not (Test-Path -LiteralPath $Pack -PathType Container)) { throw "-Data: '$Pack' is not a folder." }
    $target = (Resolve-Path -LiteralPath $Pack).ProviderPath
    if (-not (Test-Path -LiteralPath (Join-Path $target 'sql\sqlite\saffron_bank.db'))) {
        Write-Warn "$target has no sql\sqlite\saffron_bank.db; is this the B1_Smart_Data_Masking_PII folder?"
    }
    $link = Join-Path $Code 'data'
    $existing = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
    if ($existing) {
        if (-not $existing.LinkType) {
            Write-Warn "$link already exists as a real folder; leaving it as is (not linking $target)."
            return
        }
        # Delete only the link itself. Remove-Item -Recurse on a junction can
        # delete the files it points to in Windows PowerShell 5.1.
        [System.IO.Directory]::Delete($link, $false)
    }
    $type = 'SymbolicLink'
    if (Test-IsWindows) { $type = 'Junction' }
    New-Item -ItemType $type -Path $link -Target $target | Out-Null
    Write-Ok "Linked $link -> $target"
}

# Starts the pack's MySQL and PostgreSQL from the resolved pack folder, so Docker
# Compose uses the same project name as a manual "docker compose up" there and
# reuses those containers instead of starting duplicates.
function Start-PackDatabase([string]$Code) {
    $compose = Join-Path $Code 'data\docker-compose.yml'
    if (-not (Test-Path -LiteralPath $compose)) {
        Write-Warn "-StartDatabases: no docker-compose.yml in $Code\data (use -Data). Skipping."
        return
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Warn "-StartDatabases: Docker is not installed (winget install -e --id Docker.DockerDesktop). Skipping."
        return
    }
    $item = Get-Item -LiteralPath (Join-Path $Code 'data') -Force
    $pack = $item.FullName
    if ($item.LinkType) { $pack = @($item.Target)[0] }
    Write-Step 'Starting MySQL and PostgreSQL with Docker (first start loads the data, 1-3 minutes)'
    Push-Location -LiteralPath $pack
    try {
        Invoke-Native 'docker compose up' { docker compose up -d }
    } catch {
        Write-Warn 'docker compose up failed; PardaDB will use SQLite only.'
    } finally {
        Pop-Location
    }
}

# Sets up the optional AI add-on in its own venv (it needs PyTorch, which the main
# app does not). The lock pins the CUDA 12.8 build of torch; without an NVIDIA GPU
# we install the same torch version as a CPU build and drop the CUDA libraries.
function Install-AI([string]$Code, [string]$Uv, $Python) {
    $ml = Join-Path $Code 'ml'
    $lock = Join-Path $ml 'requirements.lock.txt'
    if (-not (Test-Path -LiteralPath $lock)) { throw "-WithAI: $lock not found." }
    Write-Warn 'The AI add-on downloads up to about 1 GB (PyTorch and the model libraries); more with CUDA.'

    $lines = Get-Content -LiteralPath $lock
    $hasGpu = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $hasGpu = Test-Native { nvidia-smi -L }
    }
    if ($hasGpu) {
        Write-Step 'NVIDIA GPU found: installing the CUDA build of PyTorch'
        $index = $script:TorchCudaIndex
    } else {
        Write-Step 'No NVIDIA GPU: installing the CPU build of PyTorch'
        $index = $script:TorchCpuIndex
        $lines = $lines | Where-Object { $_ -notmatch '^(nvidia-|triton==)' } |
            ForEach-Object { $_ -replace '^torch==([0-9.]+)\+.*$', 'torch==$1+cpu' }
    }
    $reqs = Join-Path ([System.IO.Path]::GetTempPath()) ("pardadb-ml-reqs-{0}.txt" -f [guid]::NewGuid())
    try {
        Set-Content -LiteralPath $reqs -Value $lines -Encoding ASCII
        $venv = Join-Path $ml '.venv'
        if ($Uv) {
            Invoke-Native 'uv venv' { & $Uv venv --quiet --allow-existing --python '>=3.10' $venv }
            # unsafe-best-match lets the other pins come from PyPI even though the
            # PyTorch index also lists some of those package names.
            Invoke-Native 'Installing the AI add-on' {
                & $Uv pip install --python (Get-VenvExe $venv 'python') --index-strategy unsafe-best-match -r $reqs --extra-index-url $index
            }
        } else {
            $exe = $Python[0]
            $pyArgs = @($Python | Select-Object -Skip 1)
            Invoke-Native 'python -m venv' { & $exe @pyArgs -m venv $venv }
            Invoke-Native 'Installing the AI add-on' {
                & (Get-VenvExe $venv 'python') -m pip install --quiet -r $reqs --extra-index-url $index
            }
        }
    } finally {
        Remove-Item -LiteralPath $reqs -Force -ErrorAction SilentlyContinue
    }
    Write-Ok "AI add-on installed in $ml\.venv (train or copy a model into ml\models, then start with: pardadb --ml)"
}

function Install-PardaDB {
    $ErrorActionPreference = 'Stop'
    # Windows PowerShell 5.1 may still default to TLS 1.0, which GitHub and astral.sh refuse.
    if ($PSVersionTable.PSEdition -eq 'Desktop') {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    }
    Write-Step "PardaDB installer (PowerShell $($PSVersionTable.PSVersion))"
    # Check -Data before any download, so a typo fails in a second, not after the install.
    if ($Data -and -not (Test-Path -LiteralPath $Data -PathType Container)) { throw "-Data: '$Data' is not a folder." }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'git not found. Install it with: winget install -e --id Git.Git   (then open a new terminal and re-run)'
    }

    # Python 3.10+ is needed. If none is found, uv can still download one, so we
    # only stop when uv is not an option either.
    $python = Find-Python
    if (-not $python) {
        # With -NoUv nothing can provide Python, so stop before cloning anything.
        if ($NoUv) { throw 'Python 3.10+ is required with -NoUv. Install it with: winget install -e --id Python.Python.3.12' }
        Write-Warn 'Python 3.10 or newer was not found; uv will provide one if it can.'
    }

    # When run as a file from inside a clone, with no -Dir or -Source, install in
    # place and leave the checkout alone. With irm | iex there is no script path.
    $repo = $null
    if ($PSCommandPath -and -not $script:DirGiven -and -not $Source -and -not $env:PARDADB_INSTALL_DIR) {
        $candidate = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
        if (Test-Path -LiteralPath (Join-Path $candidate 'FULL_CODE\pyproject.toml')) { $repo = $candidate }
    }
    $root = $Dir
    if ($repo) {
        $root = $repo
        Write-Step "Using this checkout in place: $root (not pulling)"
    } else {
        $root = [System.IO.Path]::GetFullPath($root)
        $from = $Source
        if (-not $from) { $from = $script:DefaultSource }
        Update-Checkout $from $Branch $root
    }
    $code = Join-Path $root 'FULL_CODE'
    if (-not (Test-Path -LiteralPath (Join-Path $code 'pyproject.toml'))) {
        throw "$code\pyproject.toml not found; is this the PardaDB repository?"
    }

    $uv = $null
    if (-not $NoUv) {
        $uv = Find-Uv
        if ($uv) {
            Write-Ok "uv found: $uv"
        } else {
            $uv = Install-Uv
            if ($uv) { Write-Ok "uv installed: $uv" } else { Write-Warn 'Could not install uv; falling back to python -m venv and pip.' }
        }
    }
    if (-not $uv -and -not $python) {
        throw 'No Python 3.10+ and no uv to provide one. Install Python with: winget install -e --id Python.Python.3.12'
    }

    # uv sync --frozen installs exactly what uv.lock lists and fails instead of
    # silently re-resolving to different versions.
    $venv = Join-Path $code '.venv'
    if ($uv) {
        Write-Step 'Installing PardaDB and its locked dependencies with uv'
        Push-Location -LiteralPath $code
        try { Invoke-Native 'uv sync' { & $uv sync --frozen --extra dev } } finally { Pop-Location }
    } else {
        Write-Step 'Installing PardaDB with python -m venv and pip'
        $exe = $python[0]
        $pyArgs = @($python | Select-Object -Skip 1)
        Invoke-Native 'python -m venv' { & $exe @pyArgs -m venv $venv }
        Invoke-Native 'pip upgrade' { & (Get-VenvExe $venv 'python') -m pip install --quiet --upgrade pip }
        Invoke-Native 'pip install' { & (Get-VenvExe $venv 'python') -m pip install --quiet -e "$code[dev]" }
    }
    Write-Ok "PardaDB installed in $venv"

    # .env is created once and never overwritten: after the first run it holds
    # the user's own passwords and keys.
    $envFile = Join-Path $code '.env'
    $envExample = Join-Path $code '.env.example'
    if (Test-Path -LiteralPath $envFile) {
        Write-Ok "Keeping the existing $envFile"
    } elseif (Test-Path -LiteralPath $envExample) {
        Copy-Item -LiteralPath $envExample -Destination $envFile
        Write-Ok "Created $envFile from .env.example (put the database passwords from the pack in it)"
    } else {
        Write-Warn "No .env.example found in $code; skipping .env."
    }

    if ($Data) { Set-DataLink $code $Data }
    if ($StartDatabases) { Start-PackDatabase $code }
    if ($WithAI) { Install-AI $code $uv $python }

    if ($uv) {
        $uvCmd = 'uv'
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { $uvCmd = "& '$uv'" }
        $run = "$uvCmd run pardadb"
        $test = "$uvCmd run --extra dev pytest"
    } else {
        $run = "& '$(Get-VenvExe $venv 'pardadb')'"
        $test = "& '$(Get-VenvExe $venv 'python')' -m pytest"
    }
    Write-Host ''
    Write-Host '+-- PardaDB is installed ------------------------------------------' -ForegroundColor Green
    $dataDir = Join-Path $code 'data'
    if (Test-Path -LiteralPath (Join-Path $dataDir 'sql\sqlite\saffron_bank.db')) {
        Write-Host "| 1. Problem pack: found in $dataDir"
    } else {
        Write-Host '| 1. Problem pack: copy or link B1_Smart_Data_Masking_PII to'
        Write-Host "|      $dataDir"
        Write-Host '|    (re-run with -Data C:\path\to\pack) or set PARDADB_DATASET in .env'
    }
    Write-Host "| 2. Database passwords: edit $envFile"
    Write-Host '|    (only needed for MySQL / PostgreSQL; SQLite needs nothing)'
    Write-Host "| 3. Start:  cd '$code'; $run"
    Write-Host "|    Dashboard: $script:DashboardUrl   (the terminal UI works best in Windows Terminal)"
    Write-Host "| 4. Scan only:  cd '$code'; $run scan --db sqlite"
    Write-Host "| 5. Tests:  cd '$code'; $test"
    Write-Host '+-----------------------------------------------------------------' -ForegroundColor Green
}

# Captured here because inside a function $PSBoundParameters is the function's own.
$script:DirGiven = $PSBoundParameters.ContainsKey('Dir')
Install-PardaDB
