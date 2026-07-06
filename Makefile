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

LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp
LLAMA_CPP_DIR ?= runtime/third_party/llama.cpp

BENCHMARKS_PATH ?= data/benchmarks.jsonl
MODELS_PATH ?= configs/models.example.json

NODE_NAME ?= dopey
MODEL ?= qwen2.5-coder-1.5b-q4
MODEL_PATH ?=

NODE_CLUSTER_ID ?= default
NODE_LISTEN ?= 0.0.0.0:52415
NODE_ADVERTISE_URL ?=
NODE_SEEDS ?=
NODE_DISCOVERY ?= static,mdns
NODE_MDNS_SERVICE ?= _jetsonfabric._tcp
NODE_MDNS_DOMAIN ?= local.
NODE_DATA_DIR ?= .cache/jetsonfabric
NODE_ROLE ?= auto
NODE_RUNTIME_URL ?= http://127.0.0.1:9090
NODE_ENGINE ?= jetsonfabric-runtime

RUNTIME_LISTEN ?= 127.0.0.1:9090
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

BENCH_URL ?= http://127.0.0.1:52415/v1/chat/completions
BENCH_REQUEST ?= examples/poc-local-smoke/chat-request.json
BENCH_COUNT ?= 1
BENCH_CONCURRENCY ?= 1

.PHONY: help
help:
	@printf 'JetsonFabric targets\n\n'
	@printf 'Setup:\n'
	@printf '  make setup                Clone llama.cpp if missing\n\n'
	@printf 'Build/test:\n'
	@printf '  make test                 Run Go tests\n'
	@printf '  make build                Build node binaries and CPU runtime\n'
	@printf '  make node                 Build node binaries\n'
	@printf '  make runtime              Build runtime with llama.cpp CPU\n'
	@printf '  make runtime-cuda         Build runtime with llama.cpp CUDA\n'
	@printf '  make runtime-clean        Remove runtime build directory\n\n'
	@printf 'Run:\n'
	@printf '  make run-node             Run jetsonfabric-node\n'
	@printf '  make run-runtime          Run built runtime worker\n'
	@printf '  make run-runtime-cuda     Build CUDA runtime, then run it\n\n'
	@printf 'Common knobs:\n'
	@printf '  MODEL_PATH=models/model.gguf\n'
	@printf '  RUNTIME_BUILD_JOBS=1      Safer on Jetson; try 2 or 4 if memory allows\n'
	@printf '  RUNTIME_CUDA_ARCH=87      Jetson Orin default\n'
	@printf '  CUDA_NVCC=/usr/local/cuda/bin/nvcc\n\n'
	@printf 'Developer tools:\n'
	@printf '  make bench                Run benchmark client against node API\n'
	@printf '  make clean                Remove generated build artifacts\n'

.PHONY: test
test:
	$(GO) test ./...

.PHONY: build
build: test node runtime

.PHONY: node
node:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-amd64 ./cmd/jetsonfabric-node
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-arm64 ./cmd/jetsonfabric-node

.PHONY: setup
setup:
	@if [ -f "$(LLAMA_CPP_DIR)/CMakeLists.txt" ]; then \
		printf 'llama.cpp already present at %s\n' "$(LLAMA_CPP_DIR)"; \
	else \
		mkdir -p runtime/third_party; \
		printf 'cloning llama.cpp into %s\n' "$(LLAMA_CPP_DIR)"; \
		$(GIT) clone $(LLAMA_CPP_REPO) $(LLAMA_CPP_DIR); \
	fi

.PHONY: runtime
runtime: setup
	$(CMAKE) -S runtime -B $(RUNTIME_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DJF_LLAMA_CPP_SOURCE_DIR=$(abspath $(LLAMA_CPP_DIR))
	$(CMAKE) --build $(RUNTIME_BUILD_DIR) --parallel $(RUNTIME_BUILD_JOBS)
	mkdir -p $(DIST_DIR)
	cp $(RUNTIME_BUILD_DIR)/jetsonfabric-runtime-worker $(DIST_DIR)/jetsonfabric-runtime-worker

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
	cp $(RUNTIME_BUILD_DIR)/jetsonfabric-runtime-worker $(DIST_DIR)/jetsonfabric-runtime-worker

.PHONY: runtime-clean
runtime-clean:
	rm -rf $(RUNTIME_BUILD_DIR)

.PHONY: run-node
run-node:
	$(GO) run ./cmd/jetsonfabric-node \
		--cluster-id $(NODE_CLUSTER_ID) \
		--node-name $(NODE_NAME) \
		--listen $(NODE_LISTEN) \
		--advertise-url "$(NODE_ADVERTISE_URL)" \
		--data-dir $(NODE_DATA_DIR) \
		--runtime-url $(NODE_RUNTIME_URL) \
		--engine $(NODE_ENGINE) \
		--model $(MODEL) \
		--role $(NODE_ROLE) \
		--seeds "$(NODE_SEEDS)" \
		--discovery "$(NODE_DISCOVERY)" \
		--mdns-service "$(NODE_MDNS_SERVICE)" \
		--mdns-domain "$(NODE_MDNS_DOMAIN)" \
		--benchmarks $(BENCHMARKS_PATH) \
		--models $(MODELS_PATH)

.PHONY: run-runtime
run-runtime:
	@if [ ! -x "$(DIST_DIR)/jetsonfabric-runtime-worker" ]; then \
		printf 'runtime binary missing. Run make runtime-cuda or make runtime first.\n' >&2; \
		exit 2; \
	fi
	@if [ -z "$(MODEL_PATH)" ]; then \
		printf 'MODEL_PATH is required.\n' >&2; \
		printf 'Example: make run-runtime MODEL_PATH=models/model.gguf\n' >&2; \
		exit 2; \
	fi
	./$(DIST_DIR)/jetsonfabric-runtime-worker \
		--listen $(RUNTIME_LISTEN) \
		--node-name $(NODE_NAME) \
		--engine $(RUNTIME_ENGINE) \
		--compute-backend $(RUNTIME_COMPUTE_BACKEND) \
		--model $(MODEL) \
		--model-path $(MODEL_PATH) \
		--ctx-size $(RUNTIME_CTX_SIZE) \
		--n-gpu-layers $(RUNTIME_N_GPU_LAYERS) \
		--threads $(RUNTIME_THREADS) \
		--mode $(RUNTIME_MODE) \
		--stage-index $(STAGE_INDEX) \
		--stage-count $(STAGE_COUNT) \
		--layer-start $(LAYER_START) \
		--layer-end $(LAYER_END)

.PHONY: run-runtime-cuda
run-runtime-cuda: runtime-cuda
	$(MAKE) run-runtime RUNTIME_COMPUTE_BACKEND=cuda

.PHONY: smoke-chat
smoke-chat:
	curl -sS -X POST http://127.0.0.1:52415/v1/chat/completions \
		-H 'Content-Type: application/json' \
		-d '{"model":"$(MODEL)","messages":[{"role":"user","content":"Say hello from JetsonFabric in one short sentence."}],"max_tokens":32}' | jq

.PHONY: smoke-node
smoke-node:
	curl -sS -X POST http://127.0.0.1:52415/v1/layer-split/stage \
		-H 'Content-Type: application/json' \
		-d '{"session_id":"smoke-node","request_id":"smoke-node","model_id":"$(MODEL)","stage_index":0,"node_name":"$(NODE_NAME)","role":"single","layer_start":0,"layer_end":28,"decode_step":0,"shape":[1,28,1],"dtype":"text","payload":"Say hello from JetsonFabric in one short sentence.","bytes_in":52,"transport":"http"}' | jq

.PHONY: smoke-runtime
smoke-runtime:
	curl -sS -X POST http://127.0.0.1:9090/v1/layer-split/stage \
		-H 'Content-Type: application/json' \
		-d '{"session_id":"smoke-runtime","request_id":"smoke-runtime","model_id":"$(MODEL)","stage_index":0,"node_name":"$(NODE_NAME)","role":"single","layer_start":0,"layer_end":28,"decode_step":0,"shape":[1,28,1],"dtype":"text","payload":"Say hello from JetsonFabric in one short sentence.","bytes_in":52,"transport":"http"}' | jq

.PHONY: clean
clean:
	rm -rf $(DIST_DIR) $(RUNTIME_BUILD_DIR)

.PHONY: clean-deps
clean-deps:
	rm -rf runtime/third_party/llama.cpp