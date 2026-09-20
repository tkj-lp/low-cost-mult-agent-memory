param(
  [string]$ServerUrl = "http://127.0.0.1:8765",
  [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct",
  [string[]]$Datasets = @(
    "topiocqa-factoid-ranked-1",
    "topiocqa-factoid-ranked-2",
    "topiocqa-factoid-ranked-3"
  ),
  [string[]]$Methods = @(
    "plain",
    "structured",
    "protocol-vector",
    "latent"
  ),
  [switch]$KeepExistingExperience
)

$ErrorActionPreference = "Stop"

$methods = $Methods
$datasets = $Datasets

if (-not $datasets -or $datasets.Count -eq 0) {
  throw "At least one dataset must be supplied with -Datasets."
}
$validMethods = @("plain", "structured", "protocol-vector", "latent")
$invalidMethods = @($methods | Where-Object { $_ -notin $validMethods })
if (-not $methods -or $methods.Count -eq 0) {
  throw "At least one method must be supplied with -Methods."
}
if ($invalidMethods.Count -gt 0) {
  throw "Unsupported methods: $($invalidMethods -join ', ')."
}

function New-RunConfig {
  param(
    [string]$Method,
    [string]$Dataset
  )

  return @{
    dataset = $Dataset
    taskCount = 10
    collaborationMode = $Method
    memoryMode = "hypergraph"
    retrievalMode = "hypergraph"
    backend = "local"
    localModel = $Model
  }
}

try {
  Invoke-RestMethod -Uri "$ServerUrl/api/runner/status" -TimeoutSec 10 | Out-Null
} catch {
  throw "Runner server is unavailable at $ServerUrl. Start start_local_runner.ps1 first."
}

if (-not $KeepExistingExperience) {
  Write-Host "Resetting the selected method-specific experience stores..." -ForegroundColor Yellow
  foreach ($method in $methods) {
    $config = New-RunConfig -Method $method -Dataset $datasets[0]
    $body = @{ confirm = $true; config = $config } | ConvertTo-Json -Depth 6
    $result = Invoke-RestMethod `
      -Uri "$ServerUrl/api/runner/clear-experience" `
      -Method Post `
      -ContentType "application/json" `
      -Body $body `
      -TimeoutSec 30
    Write-Host "  $method`: deleted $($result.deletedCount) experience records"
  }
}

foreach ($dataset in $datasets) {
  Write-Host "`nDataset: $dataset" -ForegroundColor Magenta
  foreach ($method in $methods) {
    $config = New-RunConfig -Method $method -Dataset $dataset
    $body = @{ config = $config } | ConvertTo-Json -Depth 6

    Write-Host "Starting $method..." -ForegroundColor Cyan
    Invoke-RestMethod `
      -Uri "$ServerUrl/api/runner/start" `
      -Method Post `
      -ContentType "application/json" `
      -Body $body `
      -TimeoutSec 30 | Out-Null

    do {
      Start-Sleep -Seconds 5
      $status = Invoke-RestMethod -Uri "$ServerUrl/api/runner/status" -TimeoutSec 30
      Write-Host "  status=$($status.status), elapsed=$($status.elapsedSeconds)s"
    } while ($status.running)

    if ($status.lastError) {
      throw "$dataset / $method failed: $($status.lastError)"
    }
    if ($status.returnCode -ne 0) {
      throw "$dataset / $method failed with exit code $($status.returnCode)"
    }

    Write-Host "Completed $dataset / $method" -ForegroundColor Green
  }
}

$runCount = $datasets.Count * $methods.Count
Write-Host "`nAll $runCount runs completed. Refresh the frontend experiment records." -ForegroundColor Green
