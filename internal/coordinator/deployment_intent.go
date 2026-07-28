package coordinator

import "github.com/SamJSui/jetsonfabric/internal/clusterplan"

type deploymentIntent struct {
	ModelID       string
	Policy        clusterplan.Policy
	ContextSize   int
	Threads       int
	NGPULayers    int
	NGPULayersSet bool
}

func intentFromSpec(spec deploymentSpec, policy clusterplan.Policy) deploymentIntent {
	intent := deploymentIntent{
		ModelID:     spec.ModelID,
		Policy:      policy,
		ContextSize: spec.ContextSize,
		Threads:     spec.Threads,
	}
	if spec.NGPULayers != nil {
		intent.NGPULayers = *spec.NGPULayers
		intent.NGPULayersSet = true
	}
	return intent
}

func (intent deploymentIntent) spec() deploymentSpec {
	spec := deploymentSpec{
		ModelID:              intent.ModelID,
		StageCount:           intent.Policy.StageCount,
		AllowColocatedStages: intent.Policy.AllowColocatedStages,
		ContextSize:          intent.ContextSize,
		Threads:              intent.Threads,
	}
	if intent.NGPULayersSet {
		value := intent.NGPULayers
		spec.NGPULayers = &value
	}
	return spec
}
