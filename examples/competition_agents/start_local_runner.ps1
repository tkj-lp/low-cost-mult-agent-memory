param(
  [switch]$UseOfficialLLMLingua2,
  [double]$LLMLingua2Rate = 0.75,
  [switch]$DisableLongTermMemory
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
$EmbeddingModel = Get-LatestSnapshotPath (Join-Path $CacheRoot "models--sentence-transformers--all-MiniLM-L6-v2")
$LLMLingua2Model = Get-LatestSnapshotPath (Join-Path $CacheRoot "models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank")

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

$env:EMBEDDING_PROVIDER = "local"
$env:LOCAL_EMBEDDING_MODEL = $EmbeddingModel
$env:HIDDEN_RETRIEVER_EMBEDDING_PROVIDER = "local"
$env:LLMLINGUA2_RATE = [string]$LLMLingua2Rate
$env:LOCAL_LLM_COMPACT_TASK = "1"
$env:LOCAL_LLM_MAX_INPUT_TOKENS = "4096"
$env:LOCAL_LLM_MAX_NEW_TOKENS = "96"
$env:LOCAL_LLM_PLANNER_MAX_NEW_TOKENS = "128"
$env:LOCAL_LLM_REASONER_MAX_NEW_TOKENS = "128"
$env:LOCAL_LLM_SUMMARIZER_MAX_NEW_TOKENS = "48"

if ($DisableLongTermMemory) {
  $env:COMPETITION_LONG_TERM_MEMORY = "0"
  Remove-Item Env:COMPETITION_MEMORY_ROOT -ErrorAction SilentlyContinue
} else {
  $env:COMPETITION_LONG_TERM_MEMORY = "1"
  $env:COMPETITION_MEMORY_ROOT = Join-Path $RepoRoot "run_result\memory_store"
}

if ($UseOfficialLLMLingua2) {
  if (-not (Test-Path $LLMLingua2Model)) {
    throw "Official LLMLingua-2 is enabled, but local model snapshot was not found."
  }
  $env:LLMLINGUA2_ENABLE_OFFICIAL = "1"
  $env:LLMLINGUA2_STRICT = "1"
  $env:LLMLINGUA2_MODEL = $LLMLingua2Model
  $env:LLMLINGUA2_DEVICE_MAP = "cuda"
  Remove-Item Env:LLMLINGUA2_DISABLE_OFFICIAL -ErrorAction SilentlyContinue
} else {
  $env:LLMLINGUA2_DISABLE_OFFICIAL = "1"
  Remove-Item Env:LLMLINGUA2_ENABLE_OFFICIAL -ErrorAction SilentlyContinue
  Remove-Item Env:LLMLINGUA2_STRICT -ErrorAction SilentlyContinue
}

Write-Host "[runner] python=$Python"
Write-Host "[runner] task model is selected on the web page"
Write-Host "[runner] LOCAL_EMBEDDING_MODEL=$env:LOCAL_EMBEDDING_MODEL"
Write-Host "[runner] official_llmlingua2=$UseOfficialLLMLingua2"
Write-Host "[runner] LLMLINGUA2_RATE=$env:LLMLINGUA2_RATE"
Write-Host "[runner] LOCAL_LLM_MAX_INPUT_TOKENS=$env:LOCAL_LLM_MAX_INPUT_TOKENS"
Write-Host "[runner] long_term_memory=$(-not $DisableLongTermMemory)"
if (-not $DisableLongTermMemory) {
  Write-Host "[runner] memory_root=$env:COMPETITION_MEMORY_ROOT"
}

& $Python examples\competition_agents\runner_server.py
