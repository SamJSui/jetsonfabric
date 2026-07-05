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
NODE_ADVERTISE_URL ?=
NODE_SEEDS ?=
NODE_DISCOVERY ?= static,mdns
NODE_MDNS_SERVICE ?= _jetsonfabric._tcp
NODE_MDNS_DOMAIN ?= local.
NODE_DATA_DIR ?= .cache/jetsonfabric
NODE_ROLE ?= auto
NODE_RUNTIME_URL ?= http://127.0.0.1:9090

ENGINE ?= jetsonfabric-runtime

RUNTIME_LISTEN ?= 127.0.0.1:9090
RUNTIME_MODE ?= data_parallel
STAGE_INDEX ?= 0
STAGE_COUNT ?= 1
LAYER_START ?= 0
LAYER_END ?= 0

BENCH_URL ?= http://127.0.0.1:52415/v1/chat/completions
BENCH_REQUEST ?= examples/poc-local-smoke/chat-request.json
BENCH_COUNT ?= 1
BENCH_CONCURRENCY ?= 1

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
	@printf '  make build                Build node binaries and runtime\n'
	@printf '  make node                 Build node binaries\n'
	@printf '  make runtime              Build C++ runtime worker\n\n'
	@printf 'Local run:\n'
	@printf '  make node-run             Run Exo-like all-in-one node locally\n'
	@printf '  make runtime-run          Run runtime locally\n\n'
	@printf 'Discovery defaults:\n'
	@printf '  NODE_DISCOVERY=static,mdns; NODE_ROLE=auto; NODE_ADVERTISE_URL may be omitted\n\n'
	@printf 'Developer tools:\n'
	@printf '  make bench                Run developer benchmark client against a node API\n\n'
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
build: test node runtime

.PHONY: node
node:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-amd64 ./cmd/jetsonfabric-node
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-node-linux-arm64 ./cmd/jetsonfabric-node

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
		--advertise-url "$(NODE_ADVERTISE_URL)" \
		--data-dir $(NODE_DATA_DIR) \
		--runtime-url $(NODE_RUNTIME_URL) \
		--engine $(ENGINE) \
		--model $(MODEL) \
		--role $(NODE_ROLE) \
		--seeds "$(NODE_SEEDS)" \
		--discovery "$(NODE_DISCOVERY)" \
		--mdns-service "$(NODE_MDNS_SERVICE)" \
		--mdns-domain "$(NODE_MDNS_DOMAIN)" \
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

.PHONY: bench
bench:
	$(GO) run ./tools/bench \
		--url $(BENCH_URL) \
		--request $(BENCH_REQUEST) \
		--count $(BENCH_COUNT) \
		--concurrency $(BENCH_CONCURRENCY)

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
