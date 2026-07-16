package node

import "github.com/SamJSui/jetsonfabric/internal/cluster"

func (a *App) memberCapabilities(base map[string]any) map[string]any {
	capabilities := make(map[string]any, len(base)+4)
	for key, value := range base {
		capabilities[key] = value
	}
	capabilities[cluster.CapabilityRuntimeStageIndex] = a.cfg.StageIndex
	capabilities[cluster.CapabilityRuntimeStageCount] = a.cfg.StageCount
	capabilities[cluster.CapabilityRuntimeLayerStart] = a.cfg.LayerStart
	capabilities[cluster.CapabilityRuntimeLayerEnd] = a.cfg.LayerEnd
	return capabilities
}
