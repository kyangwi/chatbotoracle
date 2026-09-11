# run_vllm.ps1
# Automates checking Docker GPU support and launching the vLLM Docker container
# Tailored for: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " NSSF ChatBot - vLLM Docker Launcher (16GB RTX 4060 Ti)   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check if Docker daemon is responding
Write-Host "`n[1/4] Checking Docker status..." -ForegroundColor Yellow
$dockerPing = docker version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker is not running or not responding." -ForegroundColor Red
    Write-Host "Please start 'Docker Desktop' from your Windows Start Menu and wait for it to be green." -ForegroundColor Yellow
    Exit 1
}
Write-Host "[OK] Docker Desktop is running." -ForegroundColor Green

# 2. Check GPU passthrough inside Docker
Write-Host "`n[2/4] Testing NVIDIA GPU access inside Docker..." -ForegroundColor Yellow
$gpuTest = docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Direct GPU test exited with code $LASTEXITCODE." -ForegroundColor DarkYellow
    Write-Host "Output: $gpuTest" -ForegroundColor DarkGray
    Write-Host "Will attempt to launch with standard --gpus all." -ForegroundColor Yellow
} else {
    Write-Host "[OK] GPU passthrough verified! RTX 4060 Ti accessible inside container." -ForegroundColor Green
}

# 3. Clean up any existing vLLM container
Write-Host "`n[3/4] Stopping any existing 'vllm-gpt-oss-20b' container..." -ForegroundColor Yellow
docker rm -f vllm-gpt-oss-20b 2>$null | Out-Null

# 4. Launch vLLM container
Write-Host "`n[4/4] Launching vLLM container on port 8001 (host) -> 8000 (container)..." -ForegroundColor Yellow
Write-Host "Model: gpt-oss:20b" -ForegroundColor Cyan
Write-Host "GPU Memory Utilization: 0.85 (optimized for 16GB card)" -ForegroundColor Cyan
Write-Host "Max Model Length: 8192 tokens" -ForegroundColor Cyan

$hfCachePath = "$HOME/.cache/huggingface"
if (!(Test-Path $hfCachePath)) {
    New-Item -ItemType Directory -Path $hfCachePath -Force | Out-Null
}

$modelRepo = if ($env:VLLM_HF_REPO) { $env:VLLM_HF_REPO } else { "openai/gpt-oss-20b" }
$modelName = if ($env:VLLM_MODEL) { $env:VLLM_MODEL } else { "gpt-oss:20b" }
$hfToken = if ($env:HF_TOKEN) { $env:HF_TOKEN } else { "" }

docker run -d `
    --name vllm-gpt-oss-20b `
    --gpus all `
    -p 8001:8000 `
    --ipc=host `
    --restart unless-stopped `
    -v "${hfCachePath}:/root/.cache/huggingface" `
    -e "HF_TOKEN=$hfToken" `
    -e "HUGGING_FACE_HUB_TOKEN=$hfToken" `
    vllm/vllm-openai:latest `
    --model $modelRepo `
    --served-model-name $modelName `
    --gpu-memory-utilization 0.85 `
    --max-model-len 8192 `
    --enable-auto-tool-choice `
    --tool-call-parser hermes

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] vLLM container started successfully!" -ForegroundColor Green
    Write-Host "API Endpoint: http://localhost:8001/v1" -ForegroundColor Cyan
    Write-Host "To view real-time logs, run: docker logs -f vllm-gpt-oss-20b" -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Failed to launch vLLM container. Check docker logs." -ForegroundColor Red
}
