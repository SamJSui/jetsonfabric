package cluster

import "time"

type RuntimeKind string

const (
	RuntimeKindLlamaCPP RuntimeKind = "llama.cpp"
	RuntimeKindTensorRT RuntimeKind = "tensorrt"
	RuntimeKindOllama   RuntimeKind = "ollama"
)

type RouteMode string

const (
	RouteModeSingleNode           RouteMode = "single_node"
	RouteModeReplicaBaseline      RouteMode = "replica_baseline"
	RouteModeLayerSplitExperiment RouteMode = "layer_split_experiment"
)

const (
	BackendIDLlamaLocal = "llama-local"
)

const (
	CapabilityMemoryGB     = "memory_gb"
	CapabilityAccelerators = "accelerators"
	MetricTemperatureC     = "temperature_c"
)

const (
	AcceleratorJetson = "jetson"
	AcceleratorCUDA   = "cuda"
)

type NodeRecord struct {
	NodeID       string           `json:"node_id"`
	Hostname     string           `json:"hostname"`
	Arch         string           `json:"arch"`
	OS           string           `json:"os"`
	Capabilities map[string]any   `json:"capabilities"`
	Metrics      map[string]any   `json:"metrics"`
	Backends     []RuntimeBackend `json:"backends,omitempty"`
	LastSeen     time.Time        `json:"last_seen"`
}

type HeartbeatRequest struct {
	NodeID       string           `json:"node_id"`
	Hostname     string           `json:"hostname"`
	Arch         string           `json:"arch"`
	OS           string           `json:"os"`
	Capabilities map[string]any   `json:"capabilities"`
	Metrics      map[string]any   `json:"metrics"`
	Backends     []RuntimeBackend `json:"backends,omitempty"`
}

type RuntimeBackend struct {
	ID               string      `json:"id"`
	Kind             RuntimeKind `json:"kind"`
	BaseURL          string      `json:"base_url"`
	Models           []string    `json:"models,omitempty"`
	OpenAICompatible bool        `json:"openai_compatible"`
}

type ModelProfile struct {
	ID                   string      `json:"id"`
	Family               string      `json:"family"`
	Runtime              RuntimeKind `json:"runtime"`
	MinMemoryGB          float64     `json:"min_memory_gb"`
	PreferredAccelerator *string     `json:"preferred_accelerator"`
	PlacementModes       []RouteMode `json:"placement_modes"`
}
