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

func defaultBackendFactory(backend cluster.RuntimeBackend) (runtimeclient.ChatBackend, error) {
	return runtimeclient.NewOpenAIClient(backend.BaseURL, 60*time.Second)
}

func (s *Server) selectSingleNodeBackend(model cluster.ModelProfile) (cluster.NodeRecord, cluster.RuntimeBackend, error) {
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
		for _, backend := range node.Backends {
			if backendCompatible(model, backend) {
				return node, backend, nil
			}
		}
	}
	if len(nodes) == 0 {
		return cluster.NodeRecord{}, cluster.RuntimeBackend{}, fmt.Errorf("no online nodes")
	}
	return cluster.NodeRecord{}, cluster.RuntimeBackend{}, fmt.Errorf("no compatible backend for model %q", model.ID)
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
		backend, ok := firstCompatibleBackend(model, node.Backends)
		if !ok {
			continue
		}
		candidates = append(candidates, layersplit.NodeCandidate{
			NodeName:    node.NodeName,
			BackendID:   backend.ID,
			BackendKind: backend.Kind,
			BaseURL:     backend.BaseURL,
			Weight:      layerSplitWeight(node.Capabilities),
		})
	}
	return candidates
}

func firstCompatibleBackend(model cluster.ModelProfile, backends []cluster.RuntimeBackend) (cluster.RuntimeBackend, bool) {
	for _, backend := range backends {
		if backendCompatible(model, backend) {
			return backend, true
		}
	}
	return cluster.RuntimeBackend{}, false
}

func backendCompatible(model cluster.ModelProfile, backend cluster.RuntimeBackend) bool {
	if strings.TrimSpace(backend.BaseURL) == "" {
		return false
	}
	if !backend.OpenAICompatible {
		return false
	}
	if len(backend.Models) > 0 {
		for _, modelID := range backend.Models {
			if modelID == model.ID {
				return true
			}
		}
		return false
	}
	return backend.Kind == model.Runtime
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

func layerSplitWeight(capabilities map[string]any) float64 {
	value := optionalFloat(capabilities, cluster.CapabilityLayerWeight)
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
