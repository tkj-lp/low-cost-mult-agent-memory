param(
  [string]$Model = "Qwen2.5-0.5B-Instruct",
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8765,
  [switch]$UseOfficialLLMLingua2
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = "D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Python not found: $Python"
}

function Get-LatestSnapshotPath {
  param([string]$ModelCacheDir)
  $SnapshotRoot = Join-Path $ModelCacheDir "snapshots"
  if (-not (Test-Path $SnapshotRoot)) {
    return $null
  }
  $Snapshot = Get-ChildItem $SnapshotRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($null -eq $Snapshot) {
    return $null
  }
  return $Snapshot.FullName
}

$CacheRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$Qwen05 = Get-LatestSnapshotPath (Join-Path $CacheRoot "models--Qwen--Qwen2.5-0.5B-Instruct")
$Qwen15 = Get-LatestSnapshotPath (Join-Path $CacheRoot "models--Qwen--Qwen2.5-1.5B-Instruct")
$EmbeddingModel = Get-LatestSnapshotPath (Join-Path $CacheRoot "models--sentence-transformers--all-MiniLM-L6-v2")
$LLMLingua2Model = Get-LatestSnapshotPath (Join-Path $CacheRoot "models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank")

if ($Model -in @("0.5b", "qwen0.5", "Qwen2.5-0.5B-Instruct")) {
  $LocalModel = $Qwen05
} elseif ($Model -in @("1.5b", "qwen1.5", "Qwen2.5-1.5B-Instruct")) {
  $LocalModel = $Qwen15
} else {
  $LocalModel = $Model
}

if (-not (Test-Path $LocalModel)) {
  throw "Local task model not found: $LocalModel"
}
if (-not (Test-Path $EmbeddingModel)) {
  throw "Local embedding model not found: $EmbeddingModel"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

$env:LLM_PROVIDER = "local_hf"
$env:PLANNER_LLM_PROVIDER = "local_hf"
$env:REASONER_LLM_PROVIDER = "local_hf"
$env:SUMMARIZER_LLM_PROVIDER = "local_hf"
$env:LOCAL_HIDDEN_MODEL = $LocalModel

$env:EMBEDDING_PROVIDER = "local"
$env:LOCAL_EMBEDDING_MODEL = $EmbeddingModel
$env:HIDDEN_RETRIEVER_EMBEDDING_PROVIDER = "local"

if ($UseOfficialLLMLingua2) {
  if (-not (Test-Path $LLMLingua2Model)) {
    throw "Official LLMLingua-2 is enabled, but local model snapshot was not found."
  }
  $env:LLMLINGUA2_ENABLE_OFFICIAL = "1"
  $env:LLMLINGUA2_STRICT = "1"
  $env:LLMLINGUA2_MODEL = $LLMLingua2Model
  $env:LLMLINGUA2_DEVICE_MAP = "cpu"
  Remove-Item Env:LLMLINGUA2_DISABLE_OFFICIAL -ErrorAction SilentlyContinue
} else {
  $env:LLMLINGUA2_DISABLE_OFFICIAL = "1"
  Remove-Item Env:LLMLINGUA2_ENABLE_OFFICIAL -ErrorAction SilentlyContinue
  Remove-Item Env:LLMLINGUA2_STRICT -ErrorAction SilentlyContinue
}

Write-Host "[runner] python=$Python"
Write-Host "[runner] LOCAL_HIDDEN_MODEL=$env:LOCAL_HIDDEN_MODEL"
Write-Host "[runner] LOCAL_EMBEDDING_MODEL=$env:LOCAL_EMBEDDING_MODEL"
Write-Host "[runner] official_llmlingua2=$UseOfficialLLMLingua2"

& $Python examples\competition_agents\runner_server.py --host $HostName --port $Port
