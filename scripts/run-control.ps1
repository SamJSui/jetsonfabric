param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 52415,
  [string]$JoinToken = "dev-token",
  [string]$BenchmarksPath = "data\benchmarks.jsonl"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolsGo = Join-Path (Split-Path -Parent $RepoRoot) "tools\go\bin\go.exe"
$Go = if (Test-Path $ToolsGo) { $ToolsGo } else { "go" }
$GoCache = Join-Path $RepoRoot ".cache\go-build"
New-Item -ItemType Directory -Force -Path $GoCache | Out-Null
$env:GOCACHE = $GoCache

& $Go run ./cmd/jetsonfabric-control --host $HostName --port $Port --join-token $JoinToken --benchmarks $BenchmarksPath
