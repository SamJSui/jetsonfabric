package coordinator

import (
	"cmp"
	"encoding/json"
	"slices"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
	"github.com/SamJSui/jetsonfabric/internal/routing"
)

func (s *Server) layerSplitCandidates(model cluster.ModelProfile) []layersplit.NodeCandidate {
	nodes := s.sortedNodes()
	placements := placementByNode(model, nodes)

	candidates := make([]layersplit.NodeCandidate, 0, len(nodes))
	for _, node := range nodes {
		placement := placements[node.NodeName]
		if !placement.Valid {
			continue
		}
		engine, ok := firstCompatibleEngine(model, node.Engines)
		if !ok {
			continue
		}
		candidates = append(candidates, layerSplitCandidate(node, engine))
	}
	return candidates
}

func (s *Server) sortedNodes() []cluster.NodeRecord {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)
	return nodes
}

func placementByNode(model cluster.ModelProfile, nodes []cluster.NodeRecord) map[string]routing.PlacementPreview {
	preview := routing.Preview(model, nodes)
	placements := make(map[string]routing.PlacementPreview, len(preview.Placements))
	for _, placement := range preview.Placements {
		placements[placement.NodeName] = placement
	}
	return placements
}

func layerSplitCandidate(node cluster.NodeRecord, engine cluster.EngineEndpoint) layersplit.NodeCandidate {
	return layersplit.NodeCandidate{
		NodeName:         node.NodeName,
		EngineInstanceID: engine.InstanceID,
		Engine:           engine.Engine,
		BaseURL:          engine.BaseURL,
		Weight:           pipelineWeight(node.Capabilities),
	}
}

func firstCompatibleEngine(model cluster.ModelProfile, engines []cluster.EngineEndpoint) (cluster.EngineEndpoint, bool) {
	for _, engine := range engines {
		if engineCompatible(model, engine) {
			return engine, true
		}
	}
	return cluster.EngineEndpoint{}, false
}

func engineCompatible(model cluster.ModelProfile, engine cluster.EngineEndpoint) bool {
	if strings.TrimSpace(engine.BaseURL) == "" {
		return false
	}
	if !engine.OpenAICompatible {
		return false
	}
	if len(engine.Models) > 0 {
		return engineServesModel(engine, model.ID)
	}
	return modelSupportsEngine(model, engine.Engine)
}

func engineServesModel(engine cluster.EngineEndpoint, modelID string) bool {
	for _, servedModel := range engine.Models {
		if servedModel == modelID {
			return true
		}
	}
	return false
}

func modelSupportsEngine(model cluster.ModelProfile, engine cluster.Engine) bool {
	if len(model.SupportedEngines) == 0 {
		return true
	}
	for _, supported := range model.SupportedEngines {
		if supported == engine {
			return true
		}
	}
	return false
}

func optionalFloat(values map[string]any, key string) *float64 {
	value, ok := values[key]
	if !ok {
		return nil
	}
	output, ok := numericFloat(value)
	if !ok {
		return nil
	}
	return &output
}

func numericFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func pipelineWeight(capabilities map[string]any) float64 {
	value := optionalFloat(capabilities, cluster.CapabilityPipelineWeight)
	if value == nil || *value <= 0 {
		return 1
	}
	return *value
}

func sortNodesByName(nodes []cluster.NodeRecord) {
	slices.SortFunc(nodes, func(left cluster.NodeRecord, right cluster.NodeRecord) int {
		return cmp.Compare(left.NodeName, right.NodeName)
	})
}
