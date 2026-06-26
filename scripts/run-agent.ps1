param(
  [string]$ControlUrl = "http://127.0.0.1:52415",
  [string]$JoinToken = "dev-token",
  [string]$NodeId = "dev-node",
  [string]$LlamaUrl = "",
  [string]$LlamaModels = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolsGo = Join-Path (Split-Path -Parent $RepoRoot) "tools\go\bin\go.exe"
$Go = if (Test-Path $ToolsGo) { $ToolsGo } else { "go" }
$GoCache = Join-Path $RepoRoot ".cache\go-build"
New-Item -ItemType Directory -Force -Path $GoCache | Out-Null
$env:GOCACHE = $GoCache

$Args = @(
  "run", "./cmd/jetsonfabric-agent",
  "--control-url", $ControlUrl,
  "--join-token", $JoinToken,
  "--node-id", $NodeId
)
if ($LlamaUrl -ne "") {
  $Args += @("--llama-url", $LlamaUrl)
}
if ($LlamaModels -ne "") {
  $Args += @("--llama-models", $LlamaModels)
}

& $Go @Args
