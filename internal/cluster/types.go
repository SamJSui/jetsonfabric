package cluster

// Engine identifies the inference engine hosted by a node runtime.
// It answers: "Which model implementation executes inference?"
type Engine string

const (
	EngineLlamaCPP  Engine = "llama.cpp"
	EngineSynthetic Engine = "synthetic"
)

// ExecutionMode identifies how inference work is distributed.
//
// DataParallel means each participating runtime owns a complete model replica.
// A route with one replica is still data_parallel.
//
// PipelineParallel means transformer layers are partitioned across ordered stages.
//
// TensorParallel means tensor operations such as matmuls are split across devices or nodes.
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

// OperatingSystem identifies the OS family.
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

	CapabilityRuntimeStageIndex = "runtime_stage_index"
	CapabilityRuntimeStageCount = "runtime_stage_count"
	CapabilityRuntimeLayerStart = "runtime_layer_start"
	CapabilityRuntimeLayerEnd   = "runtime_layer_end"

	// Runtime identity describes the process that is actually serving requests.
	// Host-level CUDA or engine installation detection is not a substitute for
	// these configured runtime facts.
	CapabilityRuntimeEngine           = "runtime_engine"
	CapabilityRuntimeModelID          = "runtime_model_id"
	CapabilityRuntimeModelSHA256      = "runtime_model_sha256"
	CapabilityRuntimeComputeBackend   = "runtime_compute_backend"
	CapabilityRuntimeExecutionMode    = "runtime_execution_mode"
	CapabilityRuntimeRevision         = "runtime_revision"
	CapabilityRuntimeLlamaCPPRevision = "runtime_llama_cpp_revision"
	CapabilityRuntimeCUDAActive       = "runtime_cuda_active"
	CapabilityRuntimeStartsIdle       = "runtime_starts_idle"

	MetricTemperatureC = "temperature_c"
)

// EngineEndpoint is a node-advertised endpoint for a local inference engine.
//
// BaseURL should usually be the node URL, not the raw local runtime URL, because
// cluster requests should go through the node facade and runtime gateway.
type EngineEndpoint struct {
	InstanceID       string         `json:"instance_id,omitempty"`
	Engine           Engine         `json:"engine"`
	BaseURL          string         `json:"base_url"`
	Models           []string       `json:"models,omitempty"`
	ModelSHA256      string         `json:"model_sha256,omitempty"`
	ComputeBackend   ComputeBackend `json:"compute_backend,omitempty"`
	ExecutionMode    ExecutionMode  `json:"execution_mode,omitempty"`
	OpenAICompatible bool           `json:"openai_compatible"`
}

type ModelProfile struct {
	ID               string          `json:"id"`
	Family           string          `json:"family"`
	SupportedEngines []Engine        `json:"supported_engines,omitempty"`
	LayerCount       int             `json:"layer_count,omitempty"`
	MinMemoryGB      float64         `json:"min_memory_gb"`
	PreferredCompute *ComputeBackend `json:"preferred_compute,omitempty"`
	PlacementModes   []ExecutionMode `json:"placement_modes"`
	ArtifactPath     string          `json:"artifact_path,omitempty"`
	ArtifactSHA256   string          `json:"artifact_sha256,omitempty"`
}
