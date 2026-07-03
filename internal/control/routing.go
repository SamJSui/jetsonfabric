package control

import (
	"cmp"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
	"github.com/SamJSui/jetsonfabric/internal/routing"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

func defaultEngineFactory(engine cluster.EngineEndpoint) (runtimeclient.ChatBackend, error) {
	return runtimeclient.NewOpenAIClient(engine.BaseURL, 60*time.Second)
}

func (s *Server) selectDataParallelEngine(model cluster.ModelProfile) (cluster.NodeRecord, cluster.EngineEndpoint, error) {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)

	preview := routing.Preview(model, nodes)
	placementByNode := make(map[string]routing.PlacementPreview, len(preview.Placements))
	for _, placement := range preview.Placements {
		placementByNode[placement.NodeName] = placement
	}
	for _, node := range nodes {
		placement := placementByNode[node.NodeName]
		if !placement.Valid {
			continue
		}
		for _, engine := range node.Engines {
			if engineCompatible(model, engine) {
				return node, engine, nil
			}
		}
	}
	if len(nodes) == 0 {
		return cluster.NodeRecord{}, cluster.EngineEndpoint{}, fmt.Errorf("no online nodes")
	}
	return cluster.NodeRecord{}, cluster.EngineEndpoint{}, fmt.Errorf("no compatible backend for model %q", model.ID)
}

func (s *Server) layerSplitCandidates(model cluster.ModelProfile) []layersplit.NodeCandidate {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)

	preview := routing.Preview(model, nodes)
	placementByNode := make(map[string]routing.PlacementPreview, len(preview.Placements))
	for _, placement := range preview.Placements {
		placementByNode[placement.NodeName] = placement
	}

	candidates := make([]layersplit.NodeCandidate, 0, len(nodes))
	for _, node := range nodes {
		placement := placementByNode[node.NodeName]
		if !placement.Valid {
			continue
		}
		engine, ok := firstCompatibleEngine(model, node.Engines)
		if !ok {
			continue
		}
		candidates = append(candidates, layersplit.NodeCandidate{
			NodeName:         node.NodeName,
			EngineInstanceID: engine.InstanceID,
			Engine:           engine.Engine,
			BaseURL:          engine.BaseURL,
			Weight:           pipelineWeight(node.Capabilities),
		})
	}
	return candidates
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
		for _, modelID := range engine.Models {
			if modelID == model.ID {
				return true
			}
		}
		return false
	}
	if len(model.SupportedEngines) == 0 {
		return true
	}
	for _, supported := range model.SupportedEngines {
		if supported == engine.Engine {
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
	var output float64
	switch typed := value.(type) {
	case float64:
		output = typed
	case float32:
		output = float64(typed)
	case int:
		output = float64(typed)
	case int64:
		output = float64(typed)
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return nil
		}
		output = parsed
	default:
		return nil
	}
	return &output
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
