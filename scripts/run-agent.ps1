param(
  [string]$ControlUrl = "http://127.0.0.1:52415",
  [string]$JoinToken = "dev-token",
  [string]$NodeId = "dev-node"
)

py -m jetsonmesh_agent.agent --control-url $ControlUrl --join-token $JoinToken --node-id $NodeId

