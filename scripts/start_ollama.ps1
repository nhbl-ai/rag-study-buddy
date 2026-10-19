# Run in a separate terminal; Ctrl+C stops the local model server.
param(
    [ValidateRange(1, 5)][int]$ParallelSlots = 1,
    [ValidateRange(1024, 65535)][int]$Port = 11434,
    [switch]$IntelHybrid,
    [ValidateRange(0, 15)][int]$VulkanDevice = 0,
    [ValidateRange(0, 8192)][int]$PromptCacheMiB = 512
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot 'work\ollama-runtime\ollama.exe'
$installedPath = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $runtimePath) {
    $ollamaExe = $runtimePath
} elseif ($ollamaCommand) {
    $ollamaExe = $ollamaCommand.Source
} elseif (Test-Path -LiteralPath $installedPath) {
    $ollamaExe = $installedPath
} else {
    throw 'Install Ollama for Windows first: https://ollama.com/download/windows'
}
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = [string]$ParallelSlots
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_MODELS = Join-Path $projectRoot 'work\ollama-models'
if ($IntelHybrid) {
    # Verify the selected Vulkan device is Intel in the startup log.
    # Clients must also send their measured GPU layer counts.
    $env:OLLAMA_IGPU_ENABLE = '1'
    $env:OLLAMA_VULKAN = '1'
    $env:GGML_VK_VISIBLE_DEVICES = [string]$VulkanDevice
    $env:CUDA_VISIBLE_DEVICES = '-1'
    $env:ROCR_VISIBLE_DEVICES = '-1'
    $env:HIP_VISIBLE_DEVICES = '-1'
    $env:OLLAMA_MAX_LOADED_MODELS = '2'
    # Bound the llama-server prompt cache on shared system RAM.
    $env:LLAMA_ARG_CACHE_RAM = [string]$PromptCacheMiB
}
& $ollamaExe serve
exit $LASTEXITCODE
