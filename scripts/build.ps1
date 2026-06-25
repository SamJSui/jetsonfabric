param(
  [string]$Configuration = "dev"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolsGo = Join-Path (Split-Path -Parent $RepoRoot) "tools\go\bin\go.exe"
$Go = if (Test-Path $ToolsGo) { $ToolsGo } else { "go" }
$Dist = Join-Path $RepoRoot "dist"
$GoCache = Join-Path $RepoRoot ".cache\go-build"

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
New-Item -ItemType Directory -Force -Path $GoCache | Out-Null
$env:GOCACHE = $GoCache

& $Go version
& $Go test ./...

& $Go build -o (Join-Path $Dist "jetsonmesh-control-windows-amd64.exe") ./cmd/jetsonmesh-control
& $Go build -o (Join-Path $Dist "jetsonmesh-agent-windows-amd64.exe") ./cmd/jetsonmesh-agent

$env:GOOS = "linux"
$env:GOARCH = "arm64"
& $Go build -o (Join-Path $Dist "jetsonmesh-agent-linux-arm64") ./cmd/jetsonmesh-agent
& $Go build -o (Join-Path $Dist "jetsonmesh-control-linux-arm64") ./cmd/jetsonmesh-control
Remove-Item Env:\GOOS
Remove-Item Env:\GOARCH
Remove-Item Env:\GOCACHE

Write-Host "Built JetsonMesh artifacts in $Dist"
