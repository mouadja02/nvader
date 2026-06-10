param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$VENV   = ".venv"
$PYTHON = "$PSScriptRoot\$VENV\Scripts\python"
$BIN    = "$PSScriptRoot\$VENV\Scripts"

function Invoke-Install {
    & $PYTHON -m pip install --upgrade pip
    & $PYTHON -m pip install -e ".[dev]"
}

function Invoke-Test {
    & $PYTHON -m pytest -q
}

function Invoke-Lint {
    & $PYTHON -m ruff check .
}

function Invoke-Format {
    & $PYTHON -m ruff format .
}

function Invoke-Check {
    Invoke-Lint
    Invoke-Test
}

function Invoke-Info {
    & "$BIN\nvader" info
}

function Invoke-Roadmap {
    & "$BIN\nvader" roadmap
}

function Invoke-Clean {
    if (Test-Path .pytest_cache) {
        Remove-Item -Recurse -Force .pytest_cache
    }
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
}

function Show-Help {
    Write-Host "Usage: .\make.ps1 <target>"
    Write-Host ""
    Write-Host "Targets:"
    Write-Host "  install   Install dependencies and markitdown"
    Write-Host "  test      Run pytest"
    Write-Host "  lint      Run ruff check"
    Write-Host "  format    Run ruff format"
    Write-Host "  check     Run lint then test"
    Write-Host "  info      Run nvader info"
    Write-Host "  roadmap   Run nvader roadmap"
    Write-Host "  clean     Remove build artifacts and __pycache__"
}

switch ($Target) {
    "install" { Invoke-Install }
    "test"    { Invoke-Test }
    "lint"    { Invoke-Lint }
    "format"  { Invoke-Format }
    "check"   { Invoke-Check }
    "info"    { Invoke-Info }
    "roadmap" { Invoke-Roadmap }
    "clean"   { Invoke-Clean }
    default   { Show-Help }
}
