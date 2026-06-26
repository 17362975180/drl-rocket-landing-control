param(
    [string]$RepoName = "drl-rocket-landing-control",
    [string]$Description = "Deep reinforcement learning controllers for 1D rocket soft landing",
    [ValidateSet("public", "private", "internal")]
    [string]$Visibility = "public",
    [string]$RemoteName = "origin"
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([ScriptBlock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is not installed. Install it from https://cli.github.com/ and run gh auth login."
}

Invoke-Checked { gh auth status }

$branch = (git branch --show-current).Trim()
if (-not $branch) {
    throw "No current Git branch found."
}

$status = git status --short
if ($status) {
    Write-Host "Working tree is not clean:"
    Write-Host $status
    throw "Commit or stash local changes before publishing."
}

$remoteUrl = ""
if ((git remote) -contains $RemoteName) {
    $remoteUrl = git remote get-url $RemoteName
}

if ($remoteUrl) {
    Write-Host "Remote '$RemoteName' already exists: $remoteUrl"
    Invoke-Checked { git push -u $RemoteName $branch }
    exit 0
}

$visibilityFlag = "--$Visibility"

Invoke-Checked {
    gh repo create $RepoName `
        $visibilityFlag `
        --source . `
        --remote $RemoteName `
        --push `
        --description $Description
}

Write-Host "Published $RepoName and pushed branch '$branch'."
