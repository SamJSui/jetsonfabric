package node

import "github.com/SamJSui/jetsonfabric/internal/cluster"

func (a *App) memberCapabilities(base map[string]any) map[string]any {
	capabilities := make(map[string]any, len(base)+9)
	for key, value := range base {
		capabilities[key] = value
	}
	capabilities[cluster.CapabilityRuntimeStageIndex] = a.cfg.StageIndex
	capabilities[cluster.CapabilityRuntimeStageCount] = a.cfg.StageCount
	capabilities[cluster.CapabilityRuntimeLayerStart] = a.cfg.LayerStart
	capabilities[cluster.CapabilityRuntimeLayerEnd] = a.cfg.LayerEnd
	capabilities[cluster.CapabilityRuntimeEngine] = string(a.cfg.Engine)
	capabilities[cluster.CapabilityRuntimeModelID] = a.cfg.Model
	capabilities[cluster.CapabilityRuntimeModelSHA256] = a.modelArtifactSHA256
	capabilities[cluster.CapabilityRuntimeComputeBackend] = a.cfg.RuntimeComputeBackend
	capabilities[cluster.CapabilityRuntimeExecutionMode] = a.cfg.RuntimeMode
	return capabilities
}

func (a *App) engineEndpoints() []cluster.EngineEndpoint {
	return []cluster.EngineEndpoint{{
		InstanceID:       cluster.DefaultEngineInstanceID,
		Engine:           a.cfg.Engine,
		BaseURL:          a.cfg.APIURL,
		Models:           []string{a.cfg.Model},
		ModelSHA256:      a.modelArtifactSHA256,
		ComputeBackend:   cluster.ComputeBackend(a.cfg.RuntimeComputeBackend),
		ExecutionMode:    cluster.ExecutionMode(a.cfg.RuntimeMode),
		OpenAICompatible: true,
	}}
}
