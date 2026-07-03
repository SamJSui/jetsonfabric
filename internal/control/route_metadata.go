package control

import (
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
)

func lastMessageContent(messages []chat.Message) string {
	for index := len(messages) - 1; index >= 0; index-- {
		content := strings.TrimSpace(messages[index].Content)
		if content != "" {
			return content
		}
	}
	return ""
}

func (s *Server) layerSplitRouteMetadata(plan layersplit.Plan, responses []layersplit.ActivationResponse, latency time.Duration) *chat.RouteMetadata {
	stages := make([]chat.RouteStage, 0, len(responses))
	bytesIn := 0
	bytesOut := 0
	for index, response := range responses {
		planStage := plan.Stages[index]
		bytesIn += response.BytesIn
		bytesOut += response.BytesOut
		stages = append(stages, chat.RouteStage{
			Index:       response.StageIndex,
			NodeName:    response.NodeName,
			BackendID:   planStage.BackendID,
			BackendKind: planStage.BackendKind,
			Role:        string(response.Role),
			LayerStart:  response.LayerStart,
			LayerEnd:    response.LayerEnd,
			Transport:   response.Transport,
			LatencyMS:   response.LatencyMS,
			BytesIn:     response.BytesIn,
			BytesOut:    response.BytesOut,
		})
	}
	metadata := &chat.RouteMetadata{
		Mode:      cluster.RouteModeLayerSplit,
		LatencyMS: latency.Milliseconds(),
		Stages:    stages,
		BytesIn:   bytesIn,
		BytesOut:  bytesOut,
	}
	if len(plan.Stages) > 0 {
		metadata.NodeName = plan.Stages[0].NodeName
		metadata.BackendID = plan.Stages[0].BackendID
		metadata.BackendKind = plan.Stages[0].BackendKind
	}
	return metadata
}

func stageNodeNames(responses []layersplit.ActivationResponse) []string {
	nodeNames := make([]string, 0, len(responses))
	for _, response := range responses {
		nodeNames = append(nodeNames, response.NodeName)
	}
	return nodeNames
}
