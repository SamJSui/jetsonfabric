.PHONY: help build go-build runtime-build runtime-run test smoke clean

help:
	@printf 'JetsonFabric developer targets\n\n'
	@printf '  make build          Build Go binaries and runtime stub\n'
	@printf '  make go-build       Build Go control/agent/bench binaries\n'
	@printf '  make runtime-build  Build C++ runtime worker stub\n'
	@printf '  make runtime-run    Run C++ runtime worker stub locally\n'
	@printf '  make test           Run Go tests\n'
	@printf '  make smoke          Run dopey POC smoke test with script defaults\n'
	@printf '  make clean          Remove generated build artifacts\n'

build: go-build runtime-build

go-build:
	sh scripts/build.sh

runtime-build:
	sh scripts/build-runtime.sh

runtime-run: runtime-build
	sh scripts/run-runtime-stub.sh

test:
	go test ./...

smoke:
	sh scripts/poc-dopey-smoke.sh

clean:
	rm -rf dist runtime/build .cache/go-build
