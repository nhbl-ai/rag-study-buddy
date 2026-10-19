# Run in a separate terminal; Ctrl+C stops the local model server.
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
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_MODELS = Join-Path $projectRoot 'work\ollama-models'
& $ollamaExe serve
exit $LASTEXITCODE
