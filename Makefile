SHELL := /bin/sh

LOCAL_ENV ?= .env.local
-include $(LOCAL_ENV)

GO ?= go
CMAKE ?= cmake
DOCKER ?= docker

DIST_DIR ?= dist
RUNTIME_BUILD_DIR ?= runtime/build

JOIN_TOKEN ?= dev-token
BENCHMARKS_PATH ?= data/benchmarks.jsonl
MODELS_PATH ?= configs/models.example.json

NODE_NAME ?= dopey
MODEL ?= qwen2.5-coder-1.5b-q4

NODE_CLUSTER_ID ?= default
NODE_LISTEN ?= 0.0.0.0:52415
NODE_ADVERTISE_URL ?= http://127.0.0.1:52415
NODE_SEEDS ?=
NODE_DATA_DIR ?= .cache/jetsonfabric
NODE_CONTROL_ELIGIBLE ?= true
NODE_CONTROL_PRIORITY ?= 10
NODE_RUNTIME_URL ?= http://127.0.0.1:9090

ENGINE ?= jetsonfabric-runtime

RUNTIME_LISTEN ?= 127.0.0.1:9090
RUNTIME_MODE ?= data_parallel
STAGE_INDEX ?= 0
STAGE_COUNT ?= 1
LAYER_START ?= 0
LAYER_END ?= 0

HOST ?=
EXPECTED_HOSTNAME ?= dopey

IMAGE_REPO ?= ghcr.io/samjsui
IMAGE_TAG ?= local
RUNTIME_IMAGE ?= $(IMAGE_REPO)/jetsonfabric-runtime:$(IMAGE_TAG)

.PHONY: help
help:
	@printf 'JetsonFabric targets\n\n'
	@printf 'Build/test:\n'
	@printf '  make test                 Run Go tests\n'
	@printf '  make build                Build node binaries, runtime, and bench\n'
	@printf '  make node                 Build node binaries\n'
	@printf '  make bench                Build bench binaries\n'
	@printf '  make runtime              Build C++ runtime worker\n\n'
	@printf 'Local run:\n'
	@printf '  make node-run             Run Exo-like all-in-one node locally\n'
	@printf '  make runtime-run          Run runtime locally\n\n'
	@printf 'Docker images:\n'
	@printf '  make docker-runtime       Build runtime image\n'
	@printf '  make docker-push          Push runtime image\n\n'
	@printf 'Validation/provisioning:\n'
	@printf '  make check-node HOST=...  Run Jetson readiness check\n'
	@printf '  make install-node-layout  Create /etc and /var/lib JetsonFabric dirs\n'
	@printf '  make clean                Remove generated local artifacts\n'

.PHONY: test
test:
	$(GO) test ./...

.PHONY: build
build: test node runtime bench

.PHONY: node
node:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-amd64 ./cmd/jetsonfabric-node
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-arm64 ./cmd/jetsonfabric-node

.PHONY: bench
bench:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-bench-linux-amd64 ./cmd/jetsonfabric-bench
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-bench-linux-arm64 ./cmd/jetsonfabric-bench

.PHONY: runtime
runtime:
	$(CMAKE) -S runtime -B $(RUNTIME_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(RUNTIME_BUILD_DIR) --parallel
	mkdir -p $(DIST_DIR)
	cp $(RUNTIME_BUILD_DIR)/jetsonfabric-runtime-worker $(DIST_DIR)/jetsonfabric-runtime-worker

.PHONY: node-run
node-run:
	$(GO) run ./cmd/jetsonfabric-node \
		--cluster-id $(NODE_CLUSTER_ID) \
		--node-name $(NODE_NAME) \
		--listen $(NODE_LISTEN) \
		--advertise-url $(NODE_ADVERTISE_URL) \
		--data-dir $(NODE_DATA_DIR) \
		--runtime-url $(NODE_RUNTIME_URL) \
		--engine $(ENGINE) \
		--model $(MODEL) \
		--control-eligible=$(NODE_CONTROL_ELIGIBLE) \
		--control-priority $(NODE_CONTROL_PRIORITY) \
		--seeds "$(NODE_SEEDS)" \
		--join-token $(JOIN_TOKEN) \
		--benchmarks $(BENCHMARKS_PATH) \
		--models $(MODELS_PATH)

.PHONY: runtime-run
runtime-run: runtime
	./$(DIST_DIR)/jetsonfabric-runtime-worker \
		--listen $(RUNTIME_LISTEN) \
		--node-name $(NODE_NAME) \
		--model $(MODEL) \
		--mode $(RUNTIME_MODE) \
		--stage-index $(STAGE_INDEX) \
		--stage-count $(STAGE_COUNT) \
		--layer-start $(LAYER_START) \
		--layer-end $(LAYER_END)

.PHONY: docker-runtime
docker-runtime:
	$(DOCKER) build \
		-f docker/Dockerfile.runtime \
		-t $(RUNTIME_IMAGE) \
		.

.PHONY: docker-push
docker-push:
	$(DOCKER) push $(RUNTIME_IMAGE)

.PHONY: check-node
check-node:
	@if [ -z "$(HOST)" ]; then \
		printf 'usage: make check-node HOST=dopey EXPECTED_HOSTNAME=dopey\n' >&2; \
		exit 2; \
	fi
	sh scripts/check-jetson-node.sh --host $(HOST) --expected-hostname $(EXPECTED_HOSTNAME)

.PHONY: install-node-layout
install-node-layout:
	sh scripts/install-node-layout.sh

.PHONY: clean
clean:
	rm -rf $(DIST_DIR) $(RUNTIME_BUILD_DIR)
