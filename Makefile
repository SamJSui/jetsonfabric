SHELL := /bin/sh

LOCAL_ENV ?= .env.local
-include $(LOCAL_ENV)

GO ?= go
CMAKE ?= cmake
GIT ?= git

DIST_DIR ?= dist
RUNTIME_BUILD_DIR ?= runtime/build
RUNTIME_BUILD_JOBS ?= 1
RUNTIME_CUDA_ARCH ?= 87
CUDA_NVCC ?= /usr/local/cuda/bin/nvcc
RUNTIME_BIN ?= $(DIST_DIR)/jetsonfabric-runtime-worker
NODE_BIN ?= $(DIST_DIR)/jetsonfabric-node
INTEGRATION_BUILD_DIR ?= runtime/build-integration-cpu
INTEGRATION_RUNTIME_BIN ?= $(DIST_DIR)/jetsonfabric-runtime-worker-integration-cpu

LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp
LLAMA_CPP_DIR ?= runtime/third_party/llama.cpp
LLAMA_CPP_COMMIT ?= bf2c86ddc0685f580595954056c2e77ebabfab4f

BENCHMARKS_PATH ?= data/benchmarks.jsonl
MODELS_PATH ?= configs/models.example.json

MODEL ?= qwen2.5-coder-1.5b-q4
MODEL_PATH ?=

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

# Runtime defaults used by supervised run-node and run-runtime.
RUNTIME_LISTEN ?= 127.0.0.1:0
RUNTIME_ENGINE ?= llama.cpp
RUNTIME_COMPUTE_BACKEND ?= cuda
RUNTIME_MODE ?= pipeline_parallel
RUNTIME_CTX_SIZE ?= 4096
RUNTIME_N_GPU_LAYERS ?= 999
RUNTIME_THREADS ?= 0

STAGE_INDEX ?= 0
STAGE_COUNT ?= 1
LAYER_START ?= 0
LAYER_END ?= 28

# Local development and test ports.
JF_NODE0_PORT ?= 19180
JF_NODE1_PORT ?= 19181
JF_RUNTIME_PORT ?= 19190
JF_RUNTIME1_PORT ?= 19191
JF_DEV_WORK_DIR ?= .cache/jetsonfabric/dev
DEV_NODE_URL ?= http://127.0.0.1:$(JF_NODE0_PORT)
DEV_RUNTIME_URL ?= http://127.0.0.1:$(JF_RUNTIME_PORT)
DEV_PROMPT ?= Explain JetsonFabric in one sentence.
DEV_MAX_TOKENS ?= 16

BENCH_URL ?= http://127.0.0.1:52415/v1/chat/completions
BENCH_REQUEST ?= examples/poc-local-smoke/chat-request.json
BENCH_COUNT ?= 1
BENCH_CONCURRENCY ?= 1

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
	@printf '  make node                        Build node binaries\n'
	@printf '  make runtime                     Build runtime with llama.cpp\n'
	@printf '  make runtime-cuda                Build runtime with llama.cpp + CUDA\n\n'
	@printf 'Run:\n'
	@printf '  make run-node                    Run one jetsonfabric-node in the foreground\n'
	@printf '  make run-runtime                 Run one runtime worker in the foreground\n'
	@printf '  make dev-up                      Run one full-model pipeline stage\n'
	@printf '  make dev-status                  Inspect the running development node\n'
	@printf '  make dev-chat                    Send a chat request to the development node\n'
	@printf '  make dev-kill                    Stop the recorded dev node and runtime\n'
	@printf '  make kill                        Alias for make dev-kill\n\n'
	@printf 'Developer tools:\n'
	@printf '  make bench                       Run benchmark client against node API\n'
	@printf '  make clean                       Remove generated build artifacts\n\n'
	@printf 'Common knobs:\n'
	@printf '  MODEL_PATH=models/model.gguf\n'
	@printf '  RUNTIME_BUILD_JOBS=1             Safer on Jetson; try 2 or 4 if memory allows\n'
	@printf '  RUNTIME_CUDA_ARCH=87             Jetson Orin default\n'
	@printf '  JF_NODE0_PORT=19180              Fixed local node port\n'
	@printf '  JF_RUNTIME_PORT=19190            Fixed supervised runtime port\n'
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
	JF_RUNTIME0_PORT="$(JF_RUNTIME_PORT)" \
	JF_RUNTIME1_PORT="$(JF_RUNTIME1_PORT)" \
	bash scripts/local/validate-colocated-pipeline.sh

.PHONY: build
build: test node runtime

.PHONY: node
node:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-amd64 ./cmd/jetsonfabric-node
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-arm64 ./cmd/jetsonfabric-node

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

.PHONY: run-node
run-node:
	$(GO) run ./cmd/jetsonfabric-node \
		--cluster-id "$(NODE_CLUSTER_ID)" \
		--node-name "$(NODE_NAME)" \
		--listen "$(NODE_LISTEN)" \
		--advertise-url "$(NODE_ADVERTISE_URL)" \
		--data-dir "$(NODE_DATA_DIR)" \
		--runtime-url "$(NODE_RUNTIME_URL)" \
		--runtime-bin "$(RUNTIME_BIN)" \
		--runtime-listen "$(RUNTIME_LISTEN)" \
		--runtime-compute-backend "$(RUNTIME_COMPUTE_BACKEND)" \
		--runtime-mode "$(RUNTIME_MODE)" \
		--runtime-ctx-size "$(RUNTIME_CTX_SIZE)" \
		--runtime-n-gpu-layers "$(RUNTIME_N_GPU_LAYERS)" \
		--runtime-threads "$(RUNTIME_THREADS)" \
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
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		printf 'MODEL_PATH does not exist: %s\n' "$(MODEL_PATH)" >&2; \
		printf 'Find one with: find . -type f -name "*.gguf"\n' >&2; \
		exit 2; \
	fi
	$(RUNTIME_BIN) \
		--listen "$(RUNTIME_LISTEN)" \
		--node-name "$(NODE_NAME)" \
		--engine "$(RUNTIME_ENGINE)" \
		--compute-backend "$(RUNTIME_COMPUTE_BACKEND)" \
		--model "$(MODEL)" \
		--model-path "$(MODEL_PATH)" \
		--ctx-size "$(RUNTIME_CTX_SIZE)" \
		--n-gpu-layers "$(RUNTIME_N_GPU_LAYERS)" \
		--threads "$(RUNTIME_THREADS)" \
		--mode "$(RUNTIME_MODE)" \
		--stage-index "$(STAGE_INDEX)" \
		--stage-count "$(STAGE_COUNT)" \
		--layer-start "$(LAYER_START)" \
		--layer-end "$(LAYER_END)"

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
	NODE_BIN="$(NODE_BIN)" \
	RUNTIME_COMPUTE_BACKEND="$(RUNTIME_COMPUTE_BACKEND)" \
	RUNTIME_CTX_SIZE="$(RUNTIME_CTX_SIZE)" \
	RUNTIME_N_GPU_LAYERS="$(RUNTIME_N_GPU_LAYERS)" \
	RUNTIME_THREADS="$(RUNTIME_THREADS)" \
	NODE_CLUSTER_ID="$(NODE_CLUSTER_ID)" \
	NODE_ENGINE="$(NODE_ENGINE)" \
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
		--url $(BENCH_URL) \
		--request $(BENCH_REQUEST) \
		--count $(BENCH_COUNT) \
		--concurrency $(BENCH_CONCURRENCY)

.PHONY: clean
clean:
	rm -rf $(DIST_DIR) $(RUNTIME_BUILD_DIR) $(INTEGRATION_BUILD_DIR)
