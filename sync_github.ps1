$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " Surgical Navigation GitHub Sync"
Write-Host "========================================"
Write-Host ""

$CondaExe = "C:\Users\Joshithaa\miniconda3\Scripts\conda.exe"

if (-not (Test-Path $CondaExe)) {
    Write-Host "ERROR: Conda executable not found."
    exit 1
}

Write-Host "1. Running full test suite..."
Write-Host ""

& $CondaExe run -n surgical311 python -m pytest tests/ -q

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host " TESTS FAILED - GITHUB NOT UPDATED"
    Write-Host "========================================"
    Write-Host ""
    Write-Host "Fix the failing tests before syncing."
    exit 1
}

Write-Host ""
Write-Host "All tests passed."
Write-Host ""

Write-Host "2. Checking Git repository..."

git rev-parse --is-inside-work-tree *> $null

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: This folder is not a Git repository."
    exit 1
}

Write-Host "Git repository confirmed."
Write-Host ""

Write-Host "3. Staging project changes..."

git add -A

$Changes = git status --porcelain

if (-not $Changes) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host " NOTHING NEW TO SYNC"
    Write-Host "========================================"
    Write-Host ""
    exit 0
}

Write-Host "Changes detected."
Write-Host ""

Write-Host "4. Creating checkpoint commit..."

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"

git commit -m "Project checkpoint - $Timestamp"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Git commit failed."
    exit 1
}

Write-Host ""
Write-Host "5. Pushing to GitHub..."

git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: GitHub push failed."
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host " GITHUB SYNC COMPLETE"
Write-Host "========================================"
Write-Host ""
Write-Host "Tests passed."
Write-Host "Changes committed."
Write-Host "GitHub main branch updated."
Write-Host ""