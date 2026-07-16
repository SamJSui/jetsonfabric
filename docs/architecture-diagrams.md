# Architecture Diagrams

These checked-in SVG diagrams describe the current JetsonFabric node fabric and
the activation-pipeline target. They render on GitHub without Mermaid support.

## Codebase Map

![JetsonFabric codebase map](diagrams/codebase-map.svg)

## Package Dependency View

![JetsonFabric Go package dependency diagram](diagrams/package-dependency.svg)

## Type Contract View

![JetsonFabric type contract view](diagrams/type-contract-view.svg)

## Current Node Component View

A `jetsonfabric-node` owns identity, membership, discovery, election, facade
routing, planning, runtime bridging, and its supervised runtime worker.

![JetsonFabric current node component view](diagrams/component-view.svg)

## Startup Sequence

The listener is bound before externally advertised API and runtime addresses are
derived.

![JetsonFabric node startup sequence](diagrams/startup-sequence.svg)

## Membership-Backed Planning

The planner consumes a fresh membership snapshot and returns count-aware stages,
layer ranges, topology counts, and peer API URLs.

![JetsonFabric membership-backed planning component view](diagrams/future-layer-split-component.svg)

## Model Registry and Runtime Artifact Flow

The Go node loads model metadata. The C++ runtime worker loads local model
artifacts through its configured inference engine adapter.

![JetsonFabric model registry and artifact flow](diagrams/model-artifact-flow.svg)

## Go Package Boundary View

![JetsonFabric Go package boundary view](diagrams/go-contract-view.svg)

## Node Discovery and Membership Hydration

![JetsonFabric node discovery sequence](diagrams/agent-join-sequence.svg)

## Current Any-Node Chat Path

Chat requests use `stage_index=0`, `stage_count=1`, and a text payload. Stage
position is derived from count arithmetic rather than a role string.

![JetsonFabric current any-node chat sequence](diagrams/poc-prompt-sequence.svg)

## Sequential Stage Path and Activation Target

Sequential node/runtime handoff is current. Replacing text handoff with real
activation tensors and assigned-layer execution is the next runtime milestone.

![JetsonFabric layer-split sequence](diagrams/layer-split-sequence.svg)

## Deployment View

Logical node count and physical host count are separate. Runtime URLs are local;
peer traffic uses node API URLs.

![JetsonFabric deployment view](diagrams/deployment-view.svg)

## Logical Node Identity and Topology

Multiple logical nodes may share one physical host for development. Route metadata
reports this as `topology=colocated`; physical multi-host execution reports
`topology=distributed`.

![JetsonFabric logical node identity policy](diagrams/node-name-conflict.svg)

## Test Strategy View

![JetsonFabric test strategy view](diagrams/test-strategy-view.svg)
