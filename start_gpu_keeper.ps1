# start_gpu_keeper.ps1
# Launches the persistent GPU VRAM keeper in a separate terminal window
# Keeps gpt-oss:20b loaded 24/7 in GPU memory even if Django stops or restarts.

$ScriptPath = Join-Path $PSScriptRoot "keep_gpu_loaded.py"
Write-Host "[GPU Keeper] Starting persistent GPU VRAM keeper for gpt-oss:20b..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python keep_gpu_loaded.py" -WindowStyle Normal

Write-Host "[GPU Keeper] Launched in background window. gpt-oss:20b will stay pinned in GPU VRAM." -ForegroundColor Green
