$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$envFile = Join-Path $scriptDir ".env.local"

if (-not (Test-Path $envFile)) {
    Write-Host "Missing .env.local. Create it from .env.example first:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env.local"
    Write-Host "Then edit .env.local and set LLM_API_KEY."
    exit 1
}

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) {
        return
    }
    $parts = $line.Split("=", 2)
    if ($parts.Count -ne 2) {
        return
    }
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

if (-not $env:LLM_API_KEY -or $env:LLM_API_KEY -in @("replace_with_your_api_key", "your_api_key", "你的 key")) {
    Write-Host "LLM_API_KEY is not configured in .env.local." -ForegroundColor Red
    Write-Host "Open .env.local and replace the placeholder with your real API key."
    exit 1
}

$env:PYTHONPATH = "$repoRoot\libs\langgraph;$scriptDir"

Push-Location $repoRoot
try {
    python examples/competition_agents/run_hotpot_samples.py
}
finally {
    Pop-Location
}
