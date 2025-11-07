<#
.SYNOPSIS
    RL-Bot Training Launcher - Professional Training Interface

.DESCRIPTION
    A unified PowerShell script for training the RL-Bot with a professional,
    color-coded display and easy-to-use parameters. Wraps the Python training
    script with a user-friendly interface.
    
    This script ensures proper UTF-8 encoding for all file operations to prevent
    encoding errors with YAML configuration files.

.PARAMETER Timesteps
    Total number of training timesteps (default: 10000000)

.PARAMETER Device
    Training device: 'cuda' or 'cpu' (default: auto-detect)

.PARAMETER Config
    Path to configuration file (default: configs/training_optimized.yaml)

.PARAMETER LogDir
    Directory for training logs (default: auto-generated)

.PARAMETER AerialCurriculum
    Enable aerial-focused curriculum training

.PARAMETER CurriculumStage
    Force specific curriculum stage (0-8)

.PARAMETER OfflinePretraining
    Enable offline pretraining with behavioral cloning

.PARAMETER DebugMode
    Enable debug mode with verbose logging

.PARAMETER Help
    Show this help message

.EXAMPLE
    .\train.ps1
    Start training with optimized settings

.EXAMPLE
    .\train.ps1 -Timesteps 5000000 -Device cuda -AerialCurriculum
    Train for 5M steps on GPU with aerial curriculum

.EXAMPLE
    .\train.ps1 -DebugMode -Timesteps 1000
    Quick debug run with 1000 steps

.NOTES
    Author: RL-Bot Team
    Version: 2.0.0 - Optimized for RTX 3060 with advanced PPO features
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [int]$Timesteps = 10000000,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet('cuda', 'cpu', 'auto')]
    [string]$Device = 'auto',
    
    [Parameter(Mandatory=$false)]
    [string]$Config = 'configs/training_optimized.yaml',
    
    [Parameter(Mandatory=$false)]
    [string]$LogDir = '',
    
    [Parameter(Mandatory=$false)]
    [switch]$AerialCurriculum,
    
    [Parameter(Mandatory=$false)]
    [ValidateRange(0, 8)]
    [int]$CurriculumStage = -1,
    
    [Parameter(Mandatory=$false)]
    [switch]$OfflinePretraining,
    
    [Parameter(Mandatory=$false)]
    [switch]$DebugMode,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

# Color scheme
$ColorScheme = @{
    Header = 'Cyan'
    Success = 'Green'
    Warning = 'Yellow'
    Error = 'Red'
    Info = 'White'
    Highlight = 'Magenta'
    Dim = 'DarkGray'
}

function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = 'White',
        [switch]$NoNewline
    )
    if ($NoNewline) {
        Write-Host $Text -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Text -ForegroundColor $Color
    }
}

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor $ColorScheme.Header
    Write-Host "  $Text" -ForegroundColor $ColorScheme.Header
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor $ColorScheme.Header
    Write-Host ""
}

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "───────────────────────────────────────────────────────────────────" -ForegroundColor $ColorScheme.Dim
    Write-Host "  $Text" -ForegroundColor $ColorScheme.Highlight
    Write-Host "───────────────────────────────────────────────────────────────────" -ForegroundColor $ColorScheme.Dim
}

function Write-InfoLine {
    param(
        [string]$Label,
        [string]$Value
    )
    Write-ColorText "  $Label" -Color $ColorScheme.Info -NoNewline
    Write-ColorText ": " -Color $ColorScheme.Dim -NoNewline
    Write-ColorText "$Value" -Color $ColorScheme.Success
}

function Write-Step {
    param(
        [string]$Text,
        [string]$Status = 'INFO'
    )
    $symbol = switch ($Status) {
        'SUCCESS' { '+'; $color = $ColorScheme.Success }
        'ERROR' { 'X'; $color = $ColorScheme.Error }
        'WARNING' { '!'; $color = $ColorScheme.Warning }
        default { '->'; $color = $ColorScheme.Info }
    }
    Write-ColorText "  $symbol " -Color $color -NoNewline
    Write-ColorText "$Text" -Color $ColorScheme.Info
}

function Show-Help {
    Write-Header "RL-Bot Training Launcher - Help"
    
    Write-ColorText "USAGE:" -Color $ColorScheme.Highlight
    Write-ColorText "  .\train.ps1 [OPTIONS]`n" -Color $ColorScheme.Info
    
    Write-ColorText "OPTIONS:" -Color $ColorScheme.Highlight
    Write-InfoLine "  -Timesteps <int>           " "Total training timesteps (default: 10000000)"
    Write-InfoLine "  -Device <cuda|cpu|auto>    " "Training device (default: auto)"
    Write-InfoLine "  -Config <path>             " "Configuration file (default: configs/training_optimized.yaml)"
    Write-InfoLine "  -LogDir <path>             " "Log directory (default: auto-generated)"
    Write-InfoLine "  -AerialCurriculum          " "Enable aerial curriculum"
    Write-InfoLine "  -CurriculumStage <0-8>     " "Force specific stage"
    Write-InfoLine "  -OfflinePretraining        " "Enable offline pretraining"
    Write-InfoLine "  -DebugMode                 " "Debug mode with verbose logging"
    Write-InfoLine "  -Help                      " "Show this help message`n"
    
    Write-ColorText "EXAMPLES:" -Color $ColorScheme.Highlight
    Write-ColorText "  # Basic training with optimized config" -Color $ColorScheme.Dim
    Write-ColorText "  .\train.ps1`n" -Color $ColorScheme.Info
    
    Write-ColorText "  # Custom training" -Color $ColorScheme.Dim
    Write-ColorText "  .\train.ps1 -Timesteps 5000000 -Device cuda -AerialCurriculum`n" -Color $ColorScheme.Info
    
    Write-ColorText "  # Debug mode" -Color $ColorScheme.Dim
    Write-ColorText "  .\train.ps1 -DebugMode -Timesteps 1000`n" -Color $ColorScheme.Info
    
    Write-Host ""
    Write-ColorText "NOTE: " -Color $ColorScheme.Highlight -NoNewline
    Write-ColorText "v2.0.0 uses optimized config by default with:" -Color $ColorScheme.Info
    Write-ColorText "  - Dynamic GAE lambda and entropy annealing" -Color $ColorScheme.Dim
    Write-ColorText "  - Learning rate scheduling and clip range decay" -Color $ColorScheme.Dim
    Write-ColorText "  - Reward normalization and progressive shaping" -Color $ColorScheme.Dim
    Write-ColorText "  - Automatic curriculum transitions`n" -Color $ColorScheme.Dim
    
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Set UTF-8 encoding for console output and file operations
# This prevents encoding errors with YAML files
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# Display header
Clear-Host
Write-Header "RL-Bot Training Launcher v2.0.0 - Optimized Edition"

# Check Python installation
Write-Step "Checking Python installation..." "INFO"
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Python found: $pythonVersion" "SUCCESS"
    } else {
        Write-Step "Python not found!" "ERROR"
        Write-ColorText "`nPlease install Python 3.9+ from https://www.python.org/downloads/`n" -Color $ColorScheme.Error
        exit 1
    }
} catch {
    Write-Step "Error checking Python installation" "ERROR"
    Write-ColorText "`n$($_.Exception.Message)`n" -Color $ColorScheme.Error
    exit 1
}

# Check if scripts/train.py exists
Write-Step "Checking training script..." "INFO"
if (Test-Path "scripts/train.py") {
    Write-Step "Training script found" "SUCCESS"
} else {
    Write-Step "Training script not found at scripts/train.py" "ERROR"
    Write-ColorText "`nPlease ensure you're running this from the RL-Bot directory.`n" -Color $ColorScheme.Error
    exit 1
}

# Check if config file exists
Write-Step "Checking configuration file..." "INFO"
if (Test-Path $Config) {
    Write-Step "Config found: $Config" "SUCCESS"
} else {
    Write-Step "Config file not found: $Config" "WARNING"
    Write-ColorText "  Using default configuration instead`n" -Color $ColorScheme.Warning
}

# Auto-detect device if set to auto
if ($Device -eq 'auto') {
    Write-Step "Auto-detecting training device..." "INFO"
    try {
        $cudaCheck = python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>&1
        if ($LASTEXITCODE -eq 0 -and $cudaCheck -match 'cuda|cpu') {
            $Device = $cudaCheck.Trim()
            Write-Step "Device detected: $Device" "SUCCESS"
        } else {
            $Device = 'cpu'
            Write-Step "CUDA not available, using CPU" "WARNING"
        }
    } catch {
        $Device = 'cpu'
        Write-Step "Error detecting device, defaulting to CPU" "WARNING"
    }
}

# Display training configuration
Write-Section "Training Configuration"
Write-InfoLine "Timesteps      " "$Timesteps"
Write-InfoLine "Device         " "$Device"
Write-InfoLine "Config         " "$Config"
if ($LogDir) {
    Write-InfoLine "Log Directory  " "$LogDir"
} else {
    Write-InfoLine "Log Directory  " "auto-generated"
}

if ($AerialCurriculum) {
    Write-InfoLine "Mode           " "Aerial Curriculum"
}
if ($CurriculumStage -ge 0) {
    Write-InfoLine "Forced Stage   " "$CurriculumStage"
}
if ($OfflinePretraining) {
    Write-InfoLine "Pretraining    " "Enabled"
}
if ($DebugMode) {
    Write-InfoLine "Debug Mode     " "Enabled"
}

# Build command
Write-Section "Preparing Training Command"

$pythonArgs = @(
    "scripts/train.py",
    "--config", $Config,
    "--timesteps", $Timesteps,
    "--device", $Device
)

if ($LogDir) {
    $pythonArgs += "--logdir", $LogDir
}

if ($AerialCurriculum) {
    $pythonArgs += "--aerial-curriculum"
}

if ($CurriculumStage -ge 0) {
    $pythonArgs += "--curriculum-stage", $CurriculumStage
}

if ($OfflinePretraining) {
    $pythonArgs += "--offline-pretrain"
}

if ($DebugMode) {
    $pythonArgs += "--debug"
}

# Display command
Write-Step "Command: python $($pythonArgs -join ' ')" "INFO"

# Confirmation
Write-Host ""
Write-ColorText "Ready to start training. Press " -Color $ColorScheme.Info -NoNewline
Write-ColorText "Enter" -Color $ColorScheme.Highlight -NoNewline
Write-ColorText " to continue or " -Color $ColorScheme.Info -NoNewline
Write-ColorText "Ctrl+C" -Color $ColorScheme.Warning -NoNewline
Write-ColorText " to cancel..." -Color $ColorScheme.Info
$null = Read-Host

# Start training
Write-Section "Starting Training"
Write-Host ""

$startTime = Get-Date

try {
    & python $pythonArgs
    $exitCode = $LASTEXITCODE
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host ""
    Write-Section "Training Complete"
    
    if ($exitCode -eq 0) {
        Write-Step "Training finished successfully!" "SUCCESS"
        Write-InfoLine "Duration" "$($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
        Write-Host ""
        Write-ColorText "Check the logs directory for training results and checkpoints.`n" -Color $ColorScheme.Success
    } else {
        Write-Step "Training exited with code: $exitCode" "ERROR"
        Write-InfoLine "Duration" "$($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
        Write-Host ""
        Write-ColorText "Please check the error messages above for details.`n" -Color $ColorScheme.Error
    }
    
} catch {
    Write-Host ""
    Write-Step "Error during training" "ERROR"
    Write-ColorText "`n$($_.Exception.Message)`n" -Color $ColorScheme.Error
    exit 1
}

# Footer
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor $ColorScheme.Header
Write-ColorText "  For more information, see README.md or run: " -Color $ColorScheme.Info -NoNewline
Write-ColorText ".\train.ps1 -Help" -Color $ColorScheme.Highlight
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor $ColorScheme.Header
Write-Host ""

exit $exitCode
