# Deployment invariants

This page records deployment constraints. The single-resident-slot lifecycle,
manual versioned plans, and stage-local model tensor loading are implemented.
Automatic reconciliation and non-disruptive rebalance remain target behavior.

## Ownership

```text
one physical Jetson
  -> one jetsonfabric-node
  -> one supervised runtime worker
  -> at most one resident model partition
  -> one contiguous transformer-layer range
```

The cluster owns the complete model. Each runtime owns the weights and stage-local state for only its assigned partition.

## Model admission

Inference is accepted only when the requested model matches the active cluster deployment. Requests for another model remain rejected until an explicit coordinator deployment operation changes the cluster assignment.

An inference request must not implicitly replace the active model or hide model-loading delay inside time to first token.

## Storage and memory

A model artifact may be cached and verified on local storage without occupying runtime memory. A deployment plan does not imply that a second model partition is resident in unified memory.

A runtime normally has one resident deployment slot: either empty or occupied by one deployment identity, model partition, engine instance, and stage-local session state.

## Membership and epochs

Discovery and membership expose candidate capacity. A node joining or leaving does not mutate the active deployment automatically.

Each deployment epoch has an immutable model identity, artifact identity, ordered stage placement, and layer assignment. Membership changes may produce a proposed epoch, but activation requires an explicit coordinator operation.

## Sessions and KV cache

A generation session remains pinned to the deployment epoch where it started. Every runtime owns KV state for its assigned layers.

Changing the model, artifact, or layer assignment invalidates that state. The initial lifecycle drains or fails affected sessions rather than assuming cross-epoch KV migration. Token replay and engine-specific KV migration remain future architecture decisions in the knowledge base.

## Model replacement

Without spare nodes, replacement follows this dependency order:

```text
validate target artifact and plan
  -> stop new admission
  -> finish existing sessions
  -> release current partitions
  -> load target partitions
  -> wait for all stages to report ready
  -> activate one epoch cluster-wide
  -> resume admission
```

Spare nodes may later prepare a replacement epoch while the current epoch remains active. This is a cluster-level optimization; one runtime worker still holds at most one resident partition.

## Roadmap dependency

```text
deployment identity
  -> single resident-slot state machine
  -> coordinator deployment plans
  -> admission drain and unload
  -> stage-local partition loading
  -> ready barrier
  -> atomic cluster activation
  -> lifecycle integration coverage
```
