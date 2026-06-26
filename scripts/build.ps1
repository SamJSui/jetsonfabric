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

function Invoke-Go {
  param(
    [string[]]$GoArgs
  )
  & $Go @GoArgs
  if ($LASTEXITCODE -ne 0) {
    throw "go $($GoArgs -join ' ') failed with exit code $LASTEXITCODE"
  }
}

try {
  Invoke-Go -GoArgs @("version")
  Invoke-Go -GoArgs @("test", "./...")

  Invoke-Go -GoArgs @("build", "-buildvcs=false", "-o", (Join-Path $Dist "jetsonfabric-control-windows-amd64.exe"), "./cmd/jetsonfabric-control")
  Invoke-Go -GoArgs @("build", "-buildvcs=false", "-o", (Join-Path $Dist "jetsonfabric-agent-windows-amd64.exe"), "./cmd/jetsonfabric-agent")

  $env:GOOS = "linux"
  $env:GOARCH = "arm64"
  Invoke-Go -GoArgs @("build", "-buildvcs=false", "-o", (Join-Path $Dist "jetsonfabric-agent-linux-arm64"), "./cmd/jetsonfabric-agent")
  Invoke-Go -GoArgs @("build", "-buildvcs=false", "-o", (Join-Path $Dist "jetsonfabric-control-linux-arm64"), "./cmd/jetsonfabric-control")
} finally {
  Remove-Item Env:\GOOS -ErrorAction SilentlyContinue
  Remove-Item Env:\GOARCH -ErrorAction SilentlyContinue
  Remove-Item Env:\GOCACHE -ErrorAction SilentlyContinue
}

Write-Host "Built JetsonFabric artifacts in $Dist"
