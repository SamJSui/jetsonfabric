package node

import "github.com/SamJSui/jetsonfabric/internal/cluster"

func (a *App) memberCapabilities(base map[string]any) map[string]any {
	capabilities := make(map[string]any, len(base)+13)
	for key, value := range base {
		capabilities[key] = value
	}
	if !a.cfg.RuntimeStartIdle {
		capabilities[cluster.CapabilityRuntimeStageIndex] = a.cfg.StageIndex
		capabilities[cluster.CapabilityRuntimeStageCount] = a.cfg.StageCount
		capabilities[cluster.CapabilityRuntimeLayerStart] = a.cfg.LayerStart
		capabilities[cluster.CapabilityRuntimeLayerEnd] = a.cfg.LayerEnd
		capabilities[cluster.CapabilityRuntimeModelID] = a.cfg.Model
		capabilities[cluster.CapabilityRuntimeModelSHA256] = a.modelArtifactSHA256
	}
	capabilities[cluster.CapabilityRuntimeEngine] = string(a.cfg.Engine)
	capabilities[cluster.CapabilityRuntimeComputeBackend] = a.cfg.RuntimeComputeBackend
	capabilities[cluster.CapabilityRuntimeExecutionMode] = a.cfg.RuntimeMode
	capabilities[cluster.CapabilityRuntimeRevision] = a.cfg.RuntimeRevision
	capabilities[cluster.CapabilityRuntimeLlamaCPPRevision] = a.cfg.RuntimeLlamaCPPRevision
	capabilities[cluster.CapabilityRuntimeCUDAActive] = a.cfg.RuntimeCUDAActive
	capabilities[cluster.CapabilityRuntimeStartsIdle] = a.cfg.RuntimeStartIdle
	return capabilities
}

func (a *App) engineEndpoints() []cluster.EngineEndpoint {
	endpoint := cluster.EngineEndpoint{
		InstanceID:       cluster.DefaultEngineInstanceID,
		Engine:           a.cfg.Engine,
		BaseURL:          a.cfg.APIURL,
		ComputeBackend:   cluster.ComputeBackend(a.cfg.RuntimeComputeBackend),
		ExecutionMode:    cluster.ExecutionMode(a.cfg.RuntimeMode),
		OpenAICompatible: true,
	}
	if !a.cfg.RuntimeStartIdle {
		endpoint.Models = []string{a.cfg.Model}
		endpoint.ModelSHA256 = a.modelArtifactSHA256
	}
	return []cluster.EngineEndpoint{endpoint}
}
