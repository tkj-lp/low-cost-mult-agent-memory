param([string]$PythonPath = $env:COMPETITION_PYTHON)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) {
  $Candidates = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'), (Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'))
  if ($env:CONDA_PREFIX) { $Candidates += Join-Path $env:CONDA_PREFIX 'python.exe' }
  $PythonPath = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
  if (-not $PythonPath) { $PythonPath = (Get-Command python -ErrorAction Stop).Source }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $PythonPath -m tools.verify_locked_baselines
if ($LASTEXITCODE -ne 0) { throw '固定方案校验失败，请查看变更记录。' }
& $PythonPath runner_server.py
