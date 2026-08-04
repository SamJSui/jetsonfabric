SHELL := /bin/sh

LOCAL_ENV ?= .env.local
-include $(LOCAL_ENV)

GO ?= go
CMAKE ?= cmake
GIT ?= git
PYTHON ?= python3

DIST_DIR ?= dist
RUNTIME_BUILD_DIR ?= runtime/build
RUNTIME_BUILD_JOBS ?= 1
RUNTIME_CUDA_ARCH ?= 87
CUDA_NVCC ?= /usr/local/cuda/bin/nvcc
RUNTIME_BIN ?= $(DIST_DIR)/jetsonfabric-runtime-worker
TENSOR_WORKER_BIN ?= $(DIST_DIR)/jetsonfabric-tensor-worker
NATIVE_ENGINE_BUILD_DIR ?= runtime/build-native-engine
MODEL_COMPILER_BIN ?= $(DIST_DIR)/jf-model-compile
NATIVE_MODEL_BENCH_BIN ?= $(DIST_DIR)/jf-native-model-bench
NATIVE_INFERENCE_BENCH_BIN ?= $(DIST_DIR)/jf-native-inference-bench
LLAMA_GREEDY_ORACLE_BIN ?= $(DIST_DIR)/jf-llama-greedy-oracle
NODE_BIN ?= $(DIST_DIR)/jetsonfabric-node
BENCH_BIN ?= $(DIST_DIR)/jf-bench
INTEGRATION_BUILD_DIR ?= runtime/build-integration-cpu
INTEGRATION_RUNTIME_BIN ?= $(DIST_DIR)/jetsonfabric-runtime-worker-integration-cpu

LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp
LLAMA_CPP_DIR ?= runtime/third_party/llama.cpp
LLAMA_CPP_COMMIT ?= bf2c86ddc0685f580595954056c2e77ebabfab4f
RUNTIME_REVISION ?= $(shell $(GIT) describe --always --dirty 2>/dev/null || printf unknown)
LLAMA_CPP_REVISION ?= $(LLAMA_CPP_COMMIT)
GO_REVISION_LDFLAGS := -X github.com/SamJSui/jetsonfabric/internal/node.DefaultRuntimeRevision=$(RUNTIME_REVISION) -X github.com/SamJSui/jetsonfabric/internal/node.DefaultRuntimeLlamaCPPRevision=$(LLAMA_CPP_REVISION)

BENCHMARKS_PATH ?= data/benchmarks.jsonl
MODELS_PATH ?= configs/models.example.json

MODEL ?= qwen2.5-coder-1.5b-q4
MODEL_PATH ?=
JFM_PACKAGE ?=
JFM_LAYER_START ?= 0
JFM_LAYER_END ?= 28
JFM_BENCH_ITERATIONS ?= 5
JFM_BENCH_VERIFY ?= false
JFM_BENCH_PREFETCH ?= true
JFM_BENCH_EVICT ?= false
JFM_PREFETCH_THREADS ?= 4
JFM_INFERENCE_BACKEND ?= cpu
JFM_INFERENCE_THREADS ?= 6
JFM_TOKENS ?=
JFM_MAX_TOKENS ?= 1
JFM_INFERENCE_WARMUPS ?= 1
JFM_INFERENCE_ITERATIONS ?= 3
JFM_EXPECTED_TOKENS ?=
JFM_DECODE_POLICY ?= incremental
JF_ENGINE ?= llama.cpp

# Node defaults: multi-instance safe.
NODE_NAME ?=
NODE_CLUSTER_ID ?= home-lab
NODE_LISTEN ?= 0.0.0.0:0
NODE_ADVERTISE_URL ?=
NODE_DATA_DIR ?=
NODE_RUNTIME_URL ?= auto
NODE_DISCOVERY ?= mdns
NODE_ROLE ?= auto
NODE_ENGINE ?= llama.cpp
NODE_SEEDS ?=
NODE_MDNS_SERVICE ?=
NODE_MDNS_DOMAIN ?=
JETSONFABRIC_CLUSTER_TOKEN ?= jetsonfabric-local-dev-token

# Runtime defaults used by supervised run-node and run-runtime.
RUNTIME_LISTEN ?= 127.0.0.1:0
RUNTIME_ENGINE ?= llama.cpp
RUNTIME_STAGE_TRANSPORT ?= http_binary_v1
RUNTIME_ACTIVATION_ENCODING ?= f32
RUNTIME_KV_CACHE_TYPE ?= f16
RUNTIME_COMPUTE_BACKEND ?= cuda
RUNTIME_CUDA_ACTIVE ?= true
RUNTIME_MODE ?= pipeline_parallel
RUNTIME_CTX_SIZE ?= 4096
RUNTIME_UBATCH_SIZE ?= 512
RUNTIME_N_GPU_LAYERS ?= 999
RUNTIME_THREADS ?= 0
RUNTIME_HTTP_WORKERS ?= 2
RUNTIME_PARALLEL_SESSIONS ?= 2
RUNTIME_DECODE_BATCH_SIZE ?= 1
RUNTIME_SPECULATIVE_DRAFT ?= none
RUNTIME_SPECULATIVE_MAX_TOKENS ?= 4
RUNTIME_START_IDLE ?= false
RUNTIME_TENSOR_TRANSPORT ?= llama_rpc
RUNTIME_TENSOR_RPC_PEERS ?=
RUNTIME_TENSOR_SPLIT ?=

TENSOR_WORKER_LISTEN ?= 127.0.0.1:52520
TENSOR_WORKER_CACHE_DIR ?=
TENSOR_WORKER_THREADS ?= 6
TENSOR_WORKER_ALLOW_REMOTE ?= false

STAGE_INDEX ?= 0
STAGE_COUNT ?= 1
LAYER_START ?= 0
LAYER_END ?= 28

# Local development and test ports.
JF_NODE0_PORT ?= 19180
JF_NODE1_PORT ?= 19181
JF_RUNTIME_PORT ?= 19190
JF_RUNTIME0_PORT ?= $(JF_RUNTIME_PORT)
JF_RUNTIME1_PORT ?= 19191
JF_DEV_WORK_DIR ?= .cache/jetsonfabric/dev
DEV_NODE_URL ?= http://127.0.0.1:$(JF_NODE0_PORT)
DEV_RUNTIME_URL ?= http://127.0.0.1:$(JF_RUNTIME_PORT)
DEV_PROMPT ?= Explain JetsonFabric in one sentence.
DEV_MAX_TOKENS ?= 16

BENCH_URL ?= http://127.0.0.1:52415/v1/chat/completions
BENCH_REQUEST ?= examples/chat-request.json
BENCH_COUNT ?= 1
BENCH_WARMUP ?= 0
BENCH_CONCURRENCY ?= 1
BENCH_URLS ?=
BENCH_MANIFEST ?=
BENCH_OUTPUT ?=
BENCH_STREAM ?= false
NATIVE_SCALING_MANIFEST ?= examples/native-scaling-manifest.json
NATIVE_SCALING_URL ?= http://127.0.0.1:52415
NATIVE_SCALING_OUTPUT ?= data/native-distributed-scaling.json
NATIVE_SCALING_MODELS ?=
NATIVE_SCALING_RUN_ID ?= native-scaling

.PHONY: help
help:
	@printf 'JetsonFabric targets\n\n'
	@printf 'Setup:\n'
	@printf '  make setup                       Prepare pinned llama.cpp checkout\n\n'
	@printf 'Build/test:\n'
	@printf '  make test                        Run Go unit tests\n'
	@printf '  make test-integration            Run all real-model CPU integrations\n'
	@printf '  make test-integration-single     Run one-stage real-model CPU integration\n'
	@printf '  make test-integration-pipeline   Run two-stage colocated CPU integration\n'
	@printf '  make build                       Build node binaries and runtime\n'
	@printf '  make node                        Build amd64 and arm64 node binaries\n'
	@printf '  make node-linux-arm64            Cross-compile the node for Linux ARM64\n'
	@printf '  make runtime                     Build runtime with llama.cpp\n'
	@printf '  make runtime-cuda                Build runtime with llama.cpp + CUDA\n\n'
	@printf 'Run:\n'
	@printf '  make run-node                    Run one jetsonfabric-node in the foreground\n'
	@printf '  make run-runtime                 Run one runtime worker in the foreground\n'
	@printf '  make run-tensor-worker          Expose one CUDA device to a trusted tensor driver\n'
	@printf '  make native-engine              Build and test the dependency-light native model core\n'
	@printf '  make model-compiler             Build the GGUF to JFM importer\n'
	@printf '  make model-compile              Import MODEL_PATH into JFM_PACKAGE\n'
	@printf '  make bench-native-model         Measure JFM stage mapping and prefetch\n'
	@printf '  make bench-native-inference     Measure architecture-selected native inference\n'
	@printf '  make bench-native-scaling       Run the native multi-model distributed matrix\n'
	@printf '  make dev-up                      Run one full-model pipeline stage\n'
	@printf '  make dev-status                  Inspect the running development node\n'
	@printf '  make dev-chat                    Send a chat request to the development node\n'
	@printf '  make dev-kill                    Stop the recorded dev node and runtime\n'
	@printf '  make kill                        Alias for make dev-kill\n\n'
	@printf 'Developer tools:\n'
	@printf '  make bench                       Run benchmark client against node API\n'
	@printf '  make bench BENCH_MANIFEST=...    Run a reproducible benchmark suite\n'
	@printf '  make clean                       Remove generated build artifacts\n\n'
	@printf 'Common knobs:\n'
	@printf '  MODEL_PATH=models/model.gguf      GGUF file, or JFM directory for native\n'
	@printf '  NODE_ENGINE=native                Serve a compiled JFM package\n'
	@printf '  RUNTIME_STAGE_TRANSPORT=http_binary_v1\n'
	@printf '  RUNTIME_ACTIVATION_ENCODING=f32  Use f16 to halve inter-stage payload bytes\n'
	@printf '  RUNTIME_KV_CACHE_TYPE=f16        Use q8_0 to halve llama.cpp KV cache bytes\n'
	@printf '  RUNTIME_UBATCH_SIZE=512          Bound llama.cpp physical prefill micro-batches\n'
	@printf '  RUNTIME_PARALLEL_SESSIONS=4      Reserve shared KV capacity for batching\n'
	@printf '  RUNTIME_DECODE_BATCH_SIZE=2      Enable continuous decode batching\n'
	@printf '  RUNTIME_SPECULATIVE_DRAFT=prompt_lookup  Enable target-verified lookup drafting\n'
	@printf '  RUNTIME_TENSOR_RPC_PEERS=host:52520  Enable llama.cpp tensor sharding to remote CUDA\n'
	@printf '  JFM_PACKAGE=/path/model.jfm     Canonical native model package\n'
	@printf '  JFM_LAYER_START=0 JFM_LAYER_END=14  Native stage range\n'
	@printf '  JFM_TOKENS=1,2,3               Fixed token IDs for native inference\n'
	@printf '  NATIVE_SCALING_URL=http://node:52415  Coordinator used by native scaling\n'
	@printf '  NATIVE_SCALING_MODELS=7b          Optional model labels after a cold start\n'
	@printf '  JFM_DECODE_POLICY=incremental   Native decode policy\n'
	@printf '  RUNTIME_BUILD_JOBS=1             Safer on Jetson; try 2 or 4 if memory allows\n'
	@printf '  RUNTIME_CUDA_ARCH=87             Jetson Orin default\n'
	@printf '  JF_NODE0_PORT=19180              Fixed local node port\n'
	@printf '  JF_RUNTIME_PORT=19190            Default supervised runtime port\n'
	@printf '  JF_RUNTIME0_PORT=19190           Colocated stage-0 runtime port\n'
	@printf '  JF_RUNTIME1_PORT=19191           Colocated stage-1 runtime port\n'
	@printf '  CUDA_NVCC=/usr/local/cuda/bin/nvcc\n'

.PHONY: test
test:
	$(GO) test ./...

.PHONY: test-integration
test-integration: test-integration-single test-integration-pipeline

.PHONY: test-integration-single
test-integration-single:
	@MODEL_PATH="$(MODEL_PATH)" \
	MODEL_ID="$(MODEL)" \
	RUNTIME_BUILD_DIR="$(INTEGRATION_BUILD_DIR)" \
	RUNTIME_BIN="$(INTEGRATION_RUNTIME_BIN)" \
	NODE_BIN="$(NODE_BIN)" \
	RUNTIME_BUILD_JOBS="$(RUNTIME_BUILD_JOBS)" \
	JF_NODE0_PORT="$(JF_NODE0_PORT)" \
	JF_ENGINE="$(JF_ENGINE)" \
	JF_EXPECTED_TOKENS="$(JF_EXPECTED_TOKENS)" \
	bash scripts/local/validate-single-node.sh

.PHONY: test-integration-pipeline
test-integration-pipeline:
	@MODEL_PATH="$(MODEL_PATH)" \
	MODEL_ID="$(MODEL)" \
	RUNTIME_BUILD_DIR="$(INTEGRATION_BUILD_DIR)" \
	RUNTIME_BIN="$(INTEGRATION_RUNTIME_BIN)" \
	NODE_BIN="$(NODE_BIN)" \
	RUNTIME_BUILD_JOBS="$(RUNTIME_BUILD_JOBS)" \
	JF_NODE0_PORT="$(JF_NODE0_PORT)" \
	JF_NODE1_PORT="$(JF_NODE1_PORT)" \
	JF_RUNTIME0_PORT="$(JF_RUNTIME0_PORT)" \
	JF_RUNTIME1_PORT="$(JF_RUNTIME1_PORT)" \
	JF_RUNTIME_ACTIVATION_ENCODING="$(RUNTIME_ACTIVATION_ENCODING)" \
	JF_EXPECTED_TOKENS="$(JF_EXPECTED_TOKENS)" \
	bash scripts/local/validate-colocated-pipeline.sh

.PHONY: build
build: test node runtime

.PHONY: node
node: node-linux-amd64 node-linux-arm64

.PHONY: node-linux-amd64
node-linux-amd64:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -ldflags '$(GO_REVISION_LDFLAGS)' -o $(DIST_DIR)/jetsonfabric-node-linux-amd64 ./cmd/jetsonfabric-node

.PHONY: node-linux-arm64
node-linux-arm64:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -ldflags '$(GO_REVISION_LDFLAGS)' -o $(DIST_DIR)/jetsonfabric-node-linux-arm64 ./cmd/jetsonfabric-node

.PHONY: setup
setup:
	@if [ ! -d "$(LLAMA_CPP_DIR)/.git" ]; then \
		mkdir -p runtime/third_party; \
		printf 'cloning llama.cpp into %s\n' "$(LLAMA_CPP_DIR)"; \
		$(GIT) clone --filter=blob:none $(LLAMA_CPP_REPO) $(LLAMA_CPP_DIR); \
	fi
	@printf 'preparing llama.cpp commit %s\n' "$(LLAMA_CPP_COMMIT)"
	@$(GIT) -C $(LLAMA_CPP_DIR) reset --hard
	@$(GIT) -C $(LLAMA_CPP_DIR) fetch --depth 1 origin $(LLAMA_CPP_COMMIT)
	@$(GIT) -C $(LLAMA_CPP_DIR) checkout --detach $(LLAMA_CPP_COMMIT)
	@$(GIT) -C $(LLAMA_CPP_DIR) reset --hard $(LLAMA_CPP_COMMIT)

.PHONY: runtime
runtime: setup
	$(CMAKE) -S runtime -B $(RUNTIME_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DJF_LLAMA_CPP_SOURCE_DIR=$(abspath $(LLAMA_CPP_DIR))
	$(CMAKE) --build $(RUNTIME_BUILD_DIR) --parallel $(RUNTIME_BUILD_JOBS)
	mkdir -p $(DIST_DIR)
	cp $(RUNTIME_BUILD_DIR)/jetsonfabric-runtime-worker $(RUNTIME_BIN).tmp
	chmod +x $(RUNTIME_BIN).tmp
	mv -f $(RUNTIME_BIN).tmp $(RUNTIME_BIN)

.PHONY: native-engine
native-engine:
	$(CMAKE) -S runtime/engines/native -B $(NATIVE_ENGINE_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(NATIVE_ENGINE_BUILD_DIR) --parallel $(RUNTIME_BUILD_JOBS)
	ctest --test-dir $(NATIVE_ENGINE_BUILD_DIR) --output-on-failure

.PHONY: model-compiler
model-compiler: setup
	$(CMAKE) -S runtime -B $(RUNTIME_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DJF_LLAMA_CPP_SOURCE_DIR=$(abspath $(LLAMA_CPP_DIR))
	$(CMAKE) --build $(RUNTIME_BUILD_DIR) \
		--target jf-model-compile jf-native-model-bench jf-native-inference-bench \
			jf-llama-greedy-oracle \
		--parallel $(RUNTIME_BUILD_JOBS)
	mkdir -p $(DIST_DIR)
	cp $(RUNTIME_BUILD_DIR)/jf-model-compile $(MODEL_COMPILER_BIN).tmp
	cp $(RUNTIME_BUILD_DIR)/engines/native/jf-native-model-bench $(NATIVE_MODEL_BENCH_BIN).tmp
	cp $(RUNTIME_BUILD_DIR)/engines/native/jf-native-inference-bench $(NATIVE_INFERENCE_BENCH_BIN).tmp
	cp $(RUNTIME_BUILD_DIR)/jf-llama-greedy-oracle $(LLAMA_GREEDY_ORACLE_BIN).tmp
	chmod +x $(MODEL_COMPILER_BIN).tmp $(NATIVE_MODEL_BENCH_BIN).tmp \
		$(NATIVE_INFERENCE_BENCH_BIN).tmp $(LLAMA_GREEDY_ORACLE_BIN).tmp
	mv -f $(MODEL_COMPILER_BIN).tmp $(MODEL_COMPILER_BIN)
	mv -f $(NATIVE_MODEL_BENCH_BIN).tmp $(NATIVE_MODEL_BENCH_BIN)
	mv -f $(NATIVE_INFERENCE_BENCH_BIN).tmp $(NATIVE_INFERENCE_BENCH_BIN)
	mv -f $(LLAMA_GREEDY_ORACLE_BIN).tmp $(LLAMA_GREEDY_ORACLE_BIN)

.PHONY: model-compile
model-compile: model-compiler
	@if [ -z "$(MODEL_PATH)" ] || [ -z "$(JFM_PACKAGE)" ]; then \
		printf 'MODEL_PATH and JFM_PACKAGE are required.\n' >&2; \
		exit 2; \
	fi
	$(MODEL_COMPILER_BIN) --input "$(MODEL_PATH)" --output "$(JFM_PACKAGE)"

.PHONY: bench-native-model
bench-native-model:
	@if [ ! -x "$(NATIVE_MODEL_BENCH_BIN)" ]; then \
		printf 'native model benchmark missing. Run make model-compiler.\n' >&2; \
		exit 2; \
	fi
	@if [ -z "$(JFM_PACKAGE)" ]; then \
		printf 'JFM_PACKAGE is required.\n' >&2; \
		exit 2; \
	fi
	@if [ "$(JFM_BENCH_VERIFY)" = "true" ] && [ "$(JFM_BENCH_PREFETCH)" = "true" ]; then \
		printf 'Set only one of JFM_BENCH_VERIFY or JFM_BENCH_PREFETCH; verification faults pages.\n' >&2; \
		exit 2; \
	fi
	$(NATIVE_MODEL_BENCH_BIN) \
		--package "$(JFM_PACKAGE)" \
		--layer-start "$(JFM_LAYER_START)" \
		--layer-end "$(JFM_LAYER_END)" \
		--iterations "$(JFM_BENCH_ITERATIONS)" \
		$(if $(filter true,$(JFM_BENCH_VERIFY)),--verify) \
		$(if $(filter true,$(JFM_BENCH_EVICT)),--evict) \
		--prefetch-threads "$(JFM_PREFETCH_THREADS)" \
		$(if $(filter true,$(JFM_BENCH_PREFETCH)),--prefetch)

.PHONY: bench-native-inference
bench-native-inference:
	@if [ ! -x "$(NATIVE_INFERENCE_BENCH_BIN)" ]; then \
		printf 'native inference benchmark missing. Run make model-compiler.\n' >&2; \
		exit 2; \
	fi
	@if [ -z "$(JFM_PACKAGE)" ] || [ -z "$(JFM_TOKENS)" ]; then \
		printf 'JFM_PACKAGE and JFM_TOKENS are required.\n' >&2; \
		exit 2; \
	fi
	$(NATIVE_INFERENCE_BENCH_BIN) \
		--package "$(JFM_PACKAGE)" \
		--backend "$(JFM_INFERENCE_BACKEND)" \
		--threads "$(JFM_INFERENCE_THREADS)" \
		--tokens "$(JFM_TOKENS)" \
		--max-tokens "$(JFM_MAX_TOKENS)" \
		--warmups "$(JFM_INFERENCE_WARMUPS)" \
		--iterations "$(JFM_INFERENCE_ITERATIONS)" \
		--decode-policy "$(JFM_DECODE_POLICY)" \
		$(if $(JFM_EXPECTED_TOKENS),--expected-tokens "$(JFM_EXPECTED_TOKENS)")

.PHONY: bench-native-scaling
bench-native-scaling:
	mkdir -p $(DIST_DIR)
	$(GO) build -o "$(BENCH_BIN)" ./tools/bench
	$(PYTHON) tools/bench/native_distributed_scaling.py \
		--manifest "$(NATIVE_SCALING_MANIFEST)" \
		--coordinator-url "$(NATIVE_SCALING_URL)" \
		--bench-bin "$(BENCH_BIN)" \
		--output "$(NATIVE_SCALING_OUTPUT)" \
		--run-id "$(NATIVE_SCALING_RUN_ID)" \
		$(if $(NATIVE_SCALING_MODELS),--models "$(NATIVE_SCALING_MODELS)",--continue-on-error)

.PHONY: runtime-cuda
runtime-cuda: setup
	@if [ ! -x "$(CUDA_NVCC)" ]; then \
		printf 'CUDA compiler not found at %s\n' "$(CUDA_NVCC)" >&2; \
		printf 'Set CUDA_NVCC=/path/to/nvcc or build CPU with make runtime\n' >&2; \
		exit 2; \
	fi
	$(CMAKE) -S runtime -B $(RUNTIME_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DJF_LLAMA_CPP_SOURCE_DIR=$(abspath $(LLAMA_CPP_DIR)) \
		-DGGML_CUDA=ON \
		-DCMAKE_CUDA_COMPILER=$(CUDA_NVCC) \
		-DCMAKE_CUDA_ARCHITECTURES=$(RUNTIME_CUDA_ARCH)
	$(CMAKE) --build $(RUNTIME_BUILD_DIR) --parallel $(RUNTIME_BUILD_JOBS)
	mkdir -p $(DIST_DIR)
	cp $(RUNTIME_BUILD_DIR)/jetsonfabric-runtime-worker $(RUNTIME_BIN).tmp
	chmod +x $(RUNTIME_BIN).tmp
	mv -f $(RUNTIME_BIN).tmp $(RUNTIME_BIN)
	cp $(RUNTIME_BUILD_DIR)/jetsonfabric-tensor-worker $(TENSOR_WORKER_BIN).tmp
	chmod +x $(TENSOR_WORKER_BIN).tmp
	mv -f $(TENSOR_WORKER_BIN).tmp $(TENSOR_WORKER_BIN)
	cp $(RUNTIME_BUILD_DIR)/jf-model-compile $(MODEL_COMPILER_BIN).tmp
	cp $(RUNTIME_BUILD_DIR)/engines/native/jf-native-model-bench $(NATIVE_MODEL_BENCH_BIN).tmp
	cp $(RUNTIME_BUILD_DIR)/engines/native/jf-native-inference-bench $(NATIVE_INFERENCE_BENCH_BIN).tmp
	cp $(RUNTIME_BUILD_DIR)/jf-llama-greedy-oracle $(LLAMA_GREEDY_ORACLE_BIN).tmp
	chmod +x $(MODEL_COMPILER_BIN).tmp $(NATIVE_MODEL_BENCH_BIN).tmp \
		$(NATIVE_INFERENCE_BENCH_BIN).tmp $(LLAMA_GREEDY_ORACLE_BIN).tmp
	mv -f $(MODEL_COMPILER_BIN).tmp $(MODEL_COMPILER_BIN)
	mv -f $(NATIVE_MODEL_BENCH_BIN).tmp $(NATIVE_MODEL_BENCH_BIN)
	mv -f $(NATIVE_INFERENCE_BENCH_BIN).tmp $(NATIVE_INFERENCE_BENCH_BIN)
	mv -f $(LLAMA_GREEDY_ORACLE_BIN).tmp $(LLAMA_GREEDY_ORACLE_BIN)

.PHONY: run-node
run-node:
	JETSONFABRIC_CLUSTER_TOKEN="$(JETSONFABRIC_CLUSTER_TOKEN)" $(GO) run -ldflags '$(GO_REVISION_LDFLAGS)' ./cmd/jetsonfabric-node \
		--cluster-id "$(NODE_CLUSTER_ID)" \
		--node-name "$(NODE_NAME)" \
		--listen "$(NODE_LISTEN)" \
		--advertise-url "$(NODE_ADVERTISE_URL)" \
		--data-dir "$(NODE_DATA_DIR)" \
		--runtime-url "$(NODE_RUNTIME_URL)" \
		--runtime-bin "$(RUNTIME_BIN)" \
		--runtime-listen "$(RUNTIME_LISTEN)" \
		--runtime-stage-transport "$(RUNTIME_STAGE_TRANSPORT)" \
		--runtime-activation-encoding "$(RUNTIME_ACTIVATION_ENCODING)" \
		--runtime-kv-cache-type "$(RUNTIME_KV_CACHE_TYPE)" \
		--runtime-compute-backend "$(RUNTIME_COMPUTE_BACKEND)" \
		--runtime-cuda-active="$(RUNTIME_CUDA_ACTIVE)" \
		--runtime-mode "$(RUNTIME_MODE)" \
		--runtime-ctx-size "$(RUNTIME_CTX_SIZE)" \
		--runtime-ubatch-size "$(RUNTIME_UBATCH_SIZE)" \
		--runtime-n-gpu-layers "$(RUNTIME_N_GPU_LAYERS)" \
		--runtime-threads "$(RUNTIME_THREADS)" \
		--runtime-http-workers "$(RUNTIME_HTTP_WORKERS)" \
		--runtime-parallel-sessions "$(RUNTIME_PARALLEL_SESSIONS)" \
		--runtime-decode-batch-size "$(RUNTIME_DECODE_BATCH_SIZE)" \
		--runtime-speculative-draft "$(RUNTIME_SPECULATIVE_DRAFT)" \
		--runtime-speculative-max-tokens "$(RUNTIME_SPECULATIVE_MAX_TOKENS)" \
		--runtime-idle="$(RUNTIME_START_IDLE)" \
		--engine "$(NODE_ENGINE)" \
		--model "$(MODEL)" \
		--model-path "$(MODEL_PATH)" \
		--stage-index "$(STAGE_INDEX)" \
		--stage-count "$(STAGE_COUNT)" \
		--layer-start "$(LAYER_START)" \
		--layer-end "$(LAYER_END)" \
		--role "$(NODE_ROLE)" \
		--seeds "$(NODE_SEEDS)" \
		--discovery "$(NODE_DISCOVERY)" \
		--mdns-service "$(NODE_MDNS_SERVICE)" \
		--mdns-domain "$(NODE_MDNS_DOMAIN)" \
		--benchmarks "$(BENCHMARKS_PATH)" \
		--models "$(MODELS_PATH)"

.PHONY: run-runtime
run-runtime:
	@if [ ! -x "$(RUNTIME_BIN)" ]; then \
		printf 'runtime binary missing. Run make runtime-cuda or make runtime first.\n' >&2; \
		exit 2; \
	fi
	@if [ -z "$(MODEL_PATH)" ]; then \
		printf 'MODEL_PATH is required. Example:\n' >&2; \
		printf '  make run-runtime MODEL_PATH=models/model.gguf\n' >&2; \
		exit 2; \
	fi
	@if [ ! -e "$(MODEL_PATH)" ]; then \
		printf 'MODEL_PATH does not exist: %s\n' "$(MODEL_PATH)" >&2; \
		printf 'Set MODEL_PATH to a GGUF file or JFM package directory.\n' >&2; \
		exit 2; \
	fi
	JETSONFABRIC_CLUSTER_TOKEN="$(JETSONFABRIC_CLUSTER_TOKEN)" $(RUNTIME_BIN) \
		--listen "$(RUNTIME_LISTEN)" \
		--node-name "$(NODE_NAME)" \
		--engine "$(RUNTIME_ENGINE)" \
		--stage-transport "$(RUNTIME_STAGE_TRANSPORT)" \
		--activation-encoding "$(RUNTIME_ACTIVATION_ENCODING)" \
		--kv-cache-type "$(RUNTIME_KV_CACHE_TYPE)" \
		--compute-backend "$(RUNTIME_COMPUTE_BACKEND)" \
		--model "$(MODEL)" \
		--model-path "$(MODEL_PATH)" \
		--ctx-size "$(RUNTIME_CTX_SIZE)" \
		--ubatch-size "$(RUNTIME_UBATCH_SIZE)" \
		--n-gpu-layers "$(RUNTIME_N_GPU_LAYERS)" \
		--threads "$(RUNTIME_THREADS)" \
		--http-workers "$(RUNTIME_HTTP_WORKERS)" \
		--parallel-sessions "$(RUNTIME_PARALLEL_SESSIONS)" \
		--decode-batch-size "$(RUNTIME_DECODE_BATCH_SIZE)" \
		--speculative-draft "$(RUNTIME_SPECULATIVE_DRAFT)" \
		--speculative-max-tokens "$(RUNTIME_SPECULATIVE_MAX_TOKENS)" \
		--tensor-transport "$(RUNTIME_TENSOR_TRANSPORT)" \
		--tensor-rpc-peers "$(RUNTIME_TENSOR_RPC_PEERS)" \
		--tensor-split "$(RUNTIME_TENSOR_SPLIT)" \
		--mode "$(RUNTIME_MODE)" \
		--stage-index "$(STAGE_INDEX)" \
		--stage-count "$(STAGE_COUNT)" \
		--layer-start "$(LAYER_START)" \
		--layer-end "$(LAYER_END)"

.PHONY: run-tensor-worker
run-tensor-worker:
	@if [ ! -x "$(TENSOR_WORKER_BIN)" ]; then \
		printf 'tensor worker binary missing. Run make runtime-cuda first.\n' >&2; \
		exit 2; \
	fi
	$(TENSOR_WORKER_BIN) \
		--listen "$(TENSOR_WORKER_LISTEN)" \
		--threads "$(TENSOR_WORKER_THREADS)" \
		$(if $(TENSOR_WORKER_CACHE_DIR),--cache-dir "$(TENSOR_WORKER_CACHE_DIR)") \
		$(if $(filter true,$(TENSOR_WORKER_ALLOW_REMOTE)),--allow-remote)

.PHONY: dev-up
dev-up:
	@LOCAL_ENV="$(abspath $(LOCAL_ENV))" \
	MODEL="$(MODEL)" \
	MODEL_PATH="$(MODEL_PATH)" \
	RUNTIME_BUILD_DIR="$(RUNTIME_BUILD_DIR)" \
	RUNTIME_BUILD_JOBS="$(RUNTIME_BUILD_JOBS)" \
	RUNTIME_CUDA_ARCH="$(RUNTIME_CUDA_ARCH)" \
	CUDA_NVCC="$(CUDA_NVCC)" \
	RUNTIME_BIN="$(RUNTIME_BIN)" \
	RUNTIME_ACTIVATION_ENCODING="$(RUNTIME_ACTIVATION_ENCODING)" \
	RUNTIME_KV_CACHE_TYPE="$(RUNTIME_KV_CACHE_TYPE)" \
	NODE_BIN="$(NODE_BIN)" \
	RUNTIME_COMPUTE_BACKEND="$(RUNTIME_COMPUTE_BACKEND)" \
	RUNTIME_CTX_SIZE="$(RUNTIME_CTX_SIZE)" \
	RUNTIME_UBATCH_SIZE="$(RUNTIME_UBATCH_SIZE)" \
	RUNTIME_N_GPU_LAYERS="$(RUNTIME_N_GPU_LAYERS)" \
	RUNTIME_THREADS="$(RUNTIME_THREADS)" \
	RUNTIME_HTTP_WORKERS="$(RUNTIME_HTTP_WORKERS)" \
	RUNTIME_PARALLEL_SESSIONS="$(RUNTIME_PARALLEL_SESSIONS)" \
	RUNTIME_DECODE_BATCH_SIZE="$(RUNTIME_DECODE_BATCH_SIZE)" \
	RUNTIME_SPECULATIVE_DRAFT="$(RUNTIME_SPECULATIVE_DRAFT)" \
	RUNTIME_SPECULATIVE_MAX_TOKENS="$(RUNTIME_SPECULATIVE_MAX_TOKENS)" \
	NODE_CLUSTER_ID="$(NODE_CLUSTER_ID)" \
	NODE_ENGINE="$(NODE_ENGINE)" \
	JF_CLUSTER_TOKEN="$(JETSONFABRIC_CLUSTER_TOKEN)" \
	JF_NODE0_PORT="$(JF_NODE0_PORT)" \
	JF_RUNTIME_PORT="$(JF_RUNTIME_PORT)" \
	JF_DEV_WORK_DIR="$(abspath $(JF_DEV_WORK_DIR))" \
	bash scripts/local/run-dev.sh

.PHONY: dev-kill kill
dev-kill:
	@JF_NODE0_PORT="$(JF_NODE0_PORT)" \
	JF_RUNTIME_PORT="$(JF_RUNTIME_PORT)" \
	JF_DEV_WORK_DIR="$(abspath $(JF_DEV_WORK_DIR))" \
	bash scripts/local/kill-dev.sh

kill: dev-kill

.PHONY: dev-status
dev-status:
	@printf 'Node: %s\n' "$(DEV_NODE_URL)"
	@printf 'Runtime: %s\n' "$(DEV_RUNTIME_URL)"
	@if [ -f "$(JF_DEV_WORK_DIR)/node.pid" ]; then printf 'Node PID: %s\n' "$$(cat "$(JF_DEV_WORK_DIR)/node.pid")"; fi
	@if [ -f "$(JF_DEV_WORK_DIR)/runtime.pid" ]; then printf 'Runtime PID: %s\n' "$$(cat "$(JF_DEV_WORK_DIR)/runtime.pid")"; fi
	@printf '\nHealth:\n'
	@curl -fsS "$(DEV_NODE_URL)/healthz"; printf '\n\n'
	@printf 'Members:\n'
	@curl -fsS "$(DEV_NODE_URL)/v1/cluster/members" | jq
	@printf '\nRoute preview:\n'
	@curl -fsS "$(DEV_NODE_URL)/v1/routes/preview?model=$(MODEL)&stage_count=1" | jq

.PHONY: dev-chat
dev-chat:
	@tmp="$$(mktemp)"; \
	status="$$(curl -sS -o "$$tmp" -w '%{http_code}' -X POST "$(DEV_NODE_URL)/v1/chat/completions" \
		-H 'Content-Type: application/json' \
		--data-binary "$$(jq -nc \
			--arg model "$(MODEL)" \
			--arg prompt "$(DEV_PROMPT)" \
			--argjson max_tokens "$(DEV_MAX_TOKENS)" \
			'{model:$$model,messages:[{role:"user",content:$$prompt}],max_tokens:$$max_tokens}')")"; \
	jq . "$$tmp" 2>/dev/null || cat "$$tmp"; \
	case "$$status" in 2*) result=0 ;; *) printf 'HTTP %s\n' "$$status" >&2; result=1 ;; esac; \
	rm -f "$$tmp"; \
	exit $$result

.PHONY: bench
bench:
	$(GO) run ./tools/bench \
		--url "$(BENCH_URL)" \
		--urls "$(BENCH_URLS)" \
		--request "$(BENCH_REQUEST)" \
		--count "$(BENCH_COUNT)" \
		--warmup "$(BENCH_WARMUP)" \
		--concurrency "$(BENCH_CONCURRENCY)" \
		--stream="$(BENCH_STREAM)" \
		--manifest "$(BENCH_MANIFEST)" \
		--output "$(BENCH_OUTPUT)"

.PHONY: clean
clean:
	rm -rf $(DIST_DIR) $(RUNTIME_BUILD_DIR) $(INTEGRATION_BUILD_DIR) $(NATIVE_ENGINE_BUILD_DIR)
