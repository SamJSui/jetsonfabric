param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 52415,
  [string]$JoinToken = "dev-token"
)

py -m jetsonmesh_control.server --host $HostName --port $Port --join-token $JoinToken

