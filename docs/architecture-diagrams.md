# Architecture Diagrams

These diagrams describe the intended JetsonFabric shape for P0 and the later
layer-split runtime. They are checked in as SVG files so GitHub renders them as
normal markdown images without depending on Mermaid support.

## Component View

![JetsonFabric component view](diagrams/component-view.svg)

## Go Contract View

This is a struct-level view of the main Go contracts. It is not meant to mirror
every field; it shows ownership and dependency direction.

![JetsonFabric Go contract view](diagrams/go-contract-view.svg)

## Agent Join And Heartbeat

![JetsonFabric agent join sequence](diagrams/agent-join-sequence.svg)

## P0 Prompt Path

![JetsonFabric P0 prompt sequence](diagrams/p0-prompt-sequence.svg)

## Future Layer-Split Path

In the future layer-split path, the control plane plans and observes. It should
not relay activation tensors.

![JetsonFabric future layer-split sequence](diagrams/layer-split-sequence.svg)

## Node Name Conflict Policy

![JetsonFabric node name conflict policy](diagrams/node-name-conflict.svg)

For P0, `node_name` is the identity. It defaults to the Jetson hostname, so lab
nodes can be named `dopey`, `grumpy`, and `sleepy`. Duplicate live names are
configuration conflicts rather than names the control plane silently rewrites.
