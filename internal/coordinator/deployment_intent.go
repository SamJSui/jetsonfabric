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

func intentFromSwitch(request deploymentSwitchRequest, policy clusterplan.Policy) deploymentIntent {
	intent := deploymentIntent{
		ModelID:     request.Model,
		Policy:      policy,
		ContextSize: request.ContextSize,
		Threads:     request.Threads,
	}
	if request.NGPULayers != nil {
		intent.NGPULayers = *request.NGPULayers
		intent.NGPULayersSet = true
	}
	return intent
}

func (intent deploymentIntent) switchRequest() deploymentSwitchRequest {
	request := deploymentSwitchRequest{
		Model:                intent.ModelID,
		StageCount:           intent.Policy.StageCount,
		AllowColocatedStages: intent.Policy.AllowColocatedStages,
		ContextSize:          intent.ContextSize,
		Threads:              intent.Threads,
	}
	if intent.NGPULayersSet {
		value := intent.NGPULayers
		request.NGPULayers = &value
	}
	return request
}
