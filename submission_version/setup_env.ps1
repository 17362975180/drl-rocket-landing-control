param(
    [string]$Python = "py -3.10",
    [string]$VenvPath = ".venv",
    [string]$IndexUrl = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([ScriptBlock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

# Some Windows proxy tools expose HTTPS_PROXY as https://127.0.0.1:port, which
# breaks pip/urllib3 TLS proxy negotiation. Use an HTTP proxy URL for both.
if (-not $env:HTTP_PROXY -and -not $env:HTTPS_PROXY) {
    $systemProxy = [System.Net.WebRequest]::GetSystemWebProxy().GetProxy("https://pypi.org")
    if ($systemProxy.Host -in @("127.0.0.1", "localhost")) {
        $proxyUrl = "http://$($systemProxy.Host):$($systemProxy.Port)"
        $env:HTTP_PROXY = $proxyUrl
        $env:HTTPS_PROXY = $proxyUrl
        Write-Host "Using local proxy: $proxyUrl"
    }
}

Write-Host "Creating virtual environment with $Python at $VenvPath"
Invoke-Checked { Invoke-Expression "$Python -m venv $VenvPath" }

$pythonExe = Join-Path $VenvPath "Scripts\python.exe"
Invoke-Checked { & $pythonExe -m pip install --index-url=$IndexUrl --upgrade pip }
Invoke-Checked { & $pythonExe -m pip install --index-url=$IndexUrl -r requirements.txt }

Write-Host ""
Write-Host "Environment ready."
Write-Host "Use: .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Check: python -c `"import stable_baselines3, torch; print('ok')`""
