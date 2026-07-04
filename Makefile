SHELL := /bin/sh

LOCAL_ENV ?= .env.local
-include $(LOCAL_ENV)

GO ?= go
CMAKE ?= cmake
DOCKER ?= docker
COMPOSE ?= docker compose

DIST_DIR ?= dist
RUNTIME_BUILD_DIR ?= runtime/build

CONTROL_ENV ?= .env
NODE_ENV ?= /etc/jetsonfabric/node.env

CONTROL_LISTEN ?= 127.0.0.1:52415
JOIN_TOKEN ?= dev-token
BENCHMARKS_PATH ?= data/benchmarks.jsonl
MODELS_PATH ?= configs/models.example.json

NODE_NAME ?= dopey
MODEL ?= qwen2.5-coder-1.5b-q4

CONTROL_URL ?= http://127.0.0.1:52415
AGENT_URL ?= http://127.0.0.1:52416
AGENT_LISTEN ?= 0.0.0.0:52416
AGENT_ADVERTISE_URL ?= http://127.0.0.1:52416

NODE_CLUSTER_ID ?= default
NODE_LISTEN ?= 0.0.0.0:52415
NODE_ADVERTISE_URL ?= http://127.0.0.1:52415
NODE_SEEDS ?=
NODE_DATA_DIR ?= .cache/jetsonfabric
NODE_CONTROL_ELIGIBLE ?= true
NODE_CONTROL_PRIORITY ?= 10
NODE_RUNTIME_URL ?= http://127.0.0.1:9090

ENGINE ?= jetsonfabric-runtime
ENGINE_URL ?= http://127.0.0.1:9090

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

CONTROL_IMAGE ?= $(IMAGE_REPO)/jetsonfabric-control:$(IMAGE_TAG)
AGENT_IMAGE ?= $(IMAGE_REPO)/jetsonfabric-agent:$(IMAGE_TAG)
RUNTIME_IMAGE ?= $(IMAGE_REPO)/jetsonfabric-runtime:$(IMAGE_TAG)

.PHONY: help
help:
	@printf 'JetsonFabric targets\n\n'
	@printf 'Build/test:\n'
	@printf '  make test                 Run Go tests\n'
	@printf '  make build                Build Go binaries and runtime\n'
	@printf '  make control              Build control binaries\n'
	@printf '  make agent                Build agent binaries\n'
	@printf '  make node                 Build node binaries\n'
	@printf '  make bench                Build bench binaries\n'
	@printf '  make runtime              Build C++ runtime worker\n\n'
	@printf 'Local run:\n'
	@printf '  make control-run          Run control locally via go run\n'
	@printf '  make agent-run            Run agent locally via go run\n'
	@printf '  make node-run             Run Exo-like all-in-one node locally\n'
	@printf '  make runtime-run          Run runtime locally\n\n'
	@printf 'Docker images:\n'
	@printf '  make docker-control       Build control image\n'
	@printf '  make docker-agent         Build agent image\n'
	@printf '  make docker-runtime       Build runtime image\n'
	@printf '  make docker-all           Build all local images\n'
	@printf '  make docker-push          Push all images\n\n'
	@printf 'Compose:\n'
	@printf '  make control-up           Start control Compose stack\n'
	@printf '  make control-down         Stop control Compose stack\n'
	@printf '  make control-logs         Tail control logs\n'
	@printf '  make node-up-runtime      Start node stack with JetsonFabric runtime\n'
	@printf '  make node-up-llama        Start node stack with llama.cpp engine\n'
	@printf '  make node-down            Stop node stack\n'
	@printf '  make node-logs            Tail node logs\n\n'
	@printf 'Validation/provisioning:\n'
	@printf '  make smoke                Run node smoke test\n'
	@printf '  make check-node HOST=...  Run Jetson readiness check\n'
	@printf '  make install-node-layout  Create /etc and /var/lib JetsonFabric dirs\n'
	@printf '  make clean                Remove generated local artifacts\n'

.PHONY: test
test:
	$(GO) test ./...

.PHONY: build
build: test control agent node runtime bench

.PHONY: control
control:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-control-linux-amd64 ./cmd/jetsonfabric-control
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-control-linux-arm64 ./cmd/jetsonfabric-control

.PHONY: agent
agent:
	mkdir -p $(DIST_DIR)
	GOOS=linux GOARCH=amd64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-agent-linux-amd64 ./cmd/jetsonfabric-agent
	GOOS=linux GOARCH=arm64 $(GO) build -buildvcs=false -o $(DIST_DIR)/jetsonfabric-agent-linux-arm64 ./cmd/jetsonfabric-agent

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

.PHONY: control-run
control-run:
	$(GO) run ./cmd/jetsonfabric-control \
		--listen $(CONTROL_LISTEN) \
		--join-token $(JOIN_TOKEN) \
		--benchmarks $(BENCHMARKS_PATH) \
		--models $(MODELS_PATH)

.PHONY: agent-run
agent-run:
	$(GO) run ./cmd/jetsonfabric-agent \
		--control-url $(CONTROL_URL) \
		--join-token $(JOIN_TOKEN) \
		--node-name $(NODE_NAME) \
		--listen $(AGENT_LISTEN) \
		--advertise-url $(AGENT_ADVERTISE_URL) \
		--engine $(ENGINE) \
		--engine-url $(ENGINE_URL) \
		--model $(MODEL)

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
		--control-eligible $(NODE_CONTROL_ELIGIBLE) \
		--control-priority $(NODE_CONTROL_PRIORITY) \
		--seeds $(NODE_SEEDS) \
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

.PHONY: docker-control
docker-control:
	$(DOCKER) build \
		-f docker/Dockerfile.control \
		-t $(CONTROL_IMAGE) \
		.

.PHONY: docker-agent
docker-agent:
	$(DOCKER) build \
		-f docker/Dockerfile.agent \
		-t $(AGENT_IMAGE) \
		.

.PHONY: docker-runtime
docker-runtime:
	$(DOCKER) build \
		-f docker/Dockerfile.runtime \
		-t $(RUNTIME_IMAGE) \
		.

.PHONY: docker-all
docker-all: docker-control docker-agent docker-runtime

.PHONY: docker-push
docker-push:
	$(DOCKER) push $(CONTROL_IMAGE)
	$(DOCKER) push $(AGENT_IMAGE)
	$(DOCKER) push $(RUNTIME_IMAGE)

.PHONY: control-up
control-up:
	$(COMPOSE) -f docker-compose.control.yml --env-file $(CONTROL_ENV) up -d --build

.PHONY: control-down
control-down:
	$(COMPOSE) -f docker-compose.control.yml --env-file $(CONTROL_ENV) down

.PHONY: control-logs
control-logs:
	$(COMPOSE) -f docker-compose.control.yml --env-file $(CONTROL_ENV) logs -f

.PHONY: node-up-runtime
node-up-runtime:
	$(COMPOSE) -f docker-compose.node.yml --env-file $(NODE_ENV) --profile runtime up -d --build

.PHONY: node-up-llama
node-up-llama:
	$(COMPOSE) -f docker-compose.node.yml --env-file $(NODE_ENV) --profile llama up -d

.PHONY: node-down
node-down:
	$(COMPOSE) -f docker-compose.node.yml --env-file $(NODE_ENV) down

.PHONY: node-logs
node-logs:
	$(COMPOSE) -f docker-compose.node.yml --env-file $(NODE_ENV) logs -f

.PHONY: smoke
smoke:
	CONTROL_URL=$(CONTROL_URL) \
	AGENT_URL=$(AGENT_URL) \
	NODE_NAME=$(NODE_NAME) \
	MODEL=$(MODEL) \
	EXPECTED_ENGINE=$(ENGINE) \
	EXPECTED_ROUTE_MODE=data_parallel \
	sh scripts/smoke-node.sh

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
