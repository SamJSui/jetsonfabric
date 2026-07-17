package coordinator

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

// pipelineRuntimeIdentity contains the facts that must agree for runtimes to
// exchange model activations correctly. Compute backend is intentionally not
// part of this identity: CPU and CUDA are placement/telemetry concerns as long
// as both runtimes implement the same execution and activation contract.
type pipelineRuntimeIdentity struct {
	Engine        cluster.Engine        `json:"engine"`
	ModelID       string                `json:"model_id"`
	ModelSHA256   string                `json:"model_sha256"`
	ExecutionMode cluster.ExecutionMode `json:"execution_mode"`
}

func (i pipelineRuntimeIdentity) key() string {
	return strings.Join([]string{
		string(i.Engine),
		i.ModelID,
		i.ModelSHA256,
		string(i.ExecutionMode),
	}, "|")
}

func selectPipelineRuntimeMembers(
	model cluster.ModelProfile,
	members []membership.Member,
	now time.Time,
	staleAfter time.Duration,
	requiredStages int,
) ([]membership.Member, pipelineRuntimeIdentity, error) {
	if requiredStages <= 0 {
		requiredStages = 2
	}
	groups := map[string][]membership.Member{}
	identities := map[string]pipelineRuntimeIdentity{}

	for _, member := range members {
		member = membership.Normalize(member)
		if !member.Valid() || member.IsStale(now, staleAfter) {
			continue
		}
		identity, ok := runtimeIdentityForModel(member, model)
		if !ok {
			continue
		}
		key := identity.key()
		groups[key] = append(groups[key], member)
		identities[key] = identity
	}

	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var selectedKey string
	for _, key := range keys {
		if len(groups[key]) < requiredStages {
			continue
		}
		if selectedKey == "" || len(groups[key]) > len(groups[selectedKey]) {
			selectedKey = key
		}
	}
	if selectedKey == "" {
		return nil, pipelineRuntimeIdentity{}, fmt.Errorf(
			"need %d fresh pipeline runtimes with matching engine, model artifact, and execution mode",
			requiredStages,
		)
	}
	return groups[selectedKey], identities[selectedKey], nil
}

func runtimeIdentityForModel(member membership.Member, model cluster.ModelProfile) (pipelineRuntimeIdentity, bool) {
	caps := member.Capabilities
	identity := pipelineRuntimeIdentity{
		Engine:        cluster.Engine(capabilityString(caps, cluster.CapabilityRuntimeEngine)),
		ModelID:       capabilityString(caps, cluster.CapabilityRuntimeModelID),
		ModelSHA256:   strings.ToLower(capabilityString(caps, cluster.CapabilityRuntimeModelSHA256)),
		ExecutionMode: cluster.ExecutionMode(capabilityString(caps, cluster.CapabilityRuntimeExecutionMode)),
	}
	if identity.ModelID != model.ID || identity.ModelSHA256 == "" {
		return pipelineRuntimeIdentity{}, false
	}
	if identity.ExecutionMode != cluster.ExecutionModePipelineParallel {
		return pipelineRuntimeIdentity{}, false
	}
	if !modelSupportsEngine(model, identity.Engine) {
		return pipelineRuntimeIdentity{}, false
	}
	return identity, true
}

func modelSupportsEngine(model cluster.ModelProfile, engine cluster.Engine) bool {
	for _, supported := range model.SupportedEngines {
		if supported == engine {
			return true
		}
	}
	return false
}

func capabilityString(capabilities map[string]any, key string) string {
	if capabilities == nil {
		return ""
	}
	value, ok := capabilities[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}
