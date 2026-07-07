package cluster

import "time"

// Engine identifies the local inference implementation behind a node runtime.
// It answers: "What software executes inference?"
type Engine string

const (
	EngineLlamaCPP     Engine = "llama.cpp"
	EngineJetsonFabric Engine = "jetsonfabric-runtime"
)

// ExecutionMode identifies how inference work is distributed.
//
// DataParallel means each participating node/engine owns a complete model replica.
// A one-node full-model route is data_parallel with replica_count=1.
//
// PipelineParallel means transformer layers/stages are split across stages/nodes.
//
// TensorParallel means tensor operations such as matmuls are split across devices/nodes.
type ExecutionMode string

const (
	ExecutionModeDataParallel     ExecutionMode = "data_parallel"
	ExecutionModePipelineParallel ExecutionMode = "pipeline_parallel"
	ExecutionModeTensorParallel   ExecutionMode = "tensor_parallel"
)

// DeviceClass identifies the physical platform family.
type DeviceClass string

const (
	DeviceClassUnknown DeviceClass = "unknown"
	DeviceClassJetson  DeviceClass = "jetson"
	DeviceClassLinuxPC DeviceClass = "linux_pc"
	DeviceClassMac     DeviceClass = "mac"
)

// OperatingSystem identifies the OS family reported by the node.
type OperatingSystem string

const (
	OperatingSystemUnknown OperatingSystem = "unknown"
	OperatingSystemLinux   OperatingSystem = "linux"
	OperatingSystemDarwin  OperatingSystem = "darwin"
	OperatingSystemWindows OperatingSystem = "windows"
)

// ComputeBackend identifies available local compute APIs/backends.
type ComputeBackend string

const (
	ComputeBackendCPU  ComputeBackend = "cpu"
	ComputeBackendCUDA ComputeBackend = "cuda"
)

const (
	DefaultEngineInstanceID = "default"
)

const (
	CapabilityMemoryGB        = "memory_gb"
	CapabilityDeviceClass     = "device_class"
	CapabilityComputeBackends = "compute_backends"
	CapabilityPipelineWeight  = "pipeline_weight"

	MetricTemperatureC = "temperature_c"
)

// EngineEndpoint is a node-advertised endpoint for a local inference engine.
//
// BaseURL should usually be the node URL, not the raw local runtime URL, because
// cluster requests should go through the node facade and runtime gateway.
type EngineEndpoint struct {
	InstanceID       string   `json:"instance_id,omitempty"`
	Engine           Engine   `json:"engine"`
	BaseURL          string   `json:"base_url"`
	Models           []string `json:"models,omitempty"`
	OpenAICompatible bool     `json:"openai_compatible"`
}

type NodeRecord struct {
	NodeName     string           `json:"node_name"`
	Hostname     string           `json:"hostname"`
	Arch         string           `json:"arch"`
	OS           OperatingSystem  `json:"os"`
	Capabilities map[string]any   `json:"capabilities"`
	Metrics      map[string]any   `json:"metrics"`
	Engines      []EngineEndpoint `json:"engines,omitempty"`
	LastSeen     time.Time        `json:"last_seen"`
}

type ModelProfile struct {
	ID               string          `json:"id"`
	Family           string          `json:"family"`
	SupportedEngines []Engine        `json:"supported_engines,omitempty"`
	LayerCount       int             `json:"layer_count,omitempty"`
	MinMemoryGB      float64         `json:"min_memory_gb"`
	PreferredCompute *ComputeBackend `json:"preferred_compute,omitempty"`
	PlacementModes   []ExecutionMode `json:"placement_modes"`
}
