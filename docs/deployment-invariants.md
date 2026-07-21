# Deployment invariants

This page records the implemented deployment and rebalance constraints.

## Ownership

```text
one physical Jetson
  -> one jetsonfabric-node
  -> one supervised runtime worker
  -> one partition per resident deployment epoch
  -> one contiguous transformer-layer range per epoch
```

The cluster owns the complete model. Each runtime owns the weights and
stage-local state for its assigned range. During rebalance, a runtime may hold an
old and a replacement epoch at the same time; each remains a separate engine and
session namespace.

## Model admission

Only the coordinator's active epoch accepts new requests. An admission copies
that immutable plan and increments its epoch-specific in-flight count. Loading
or activating another model never happens implicitly inside inference.

Requests admitted before publication keep the old plan. Requests admitted after
publication get the new plan. The runtime accepts stage work for both `active`
and `draining` identities so old KV state remains usable until release.

## Storage and memory

A verified model artifact on local storage is not resident model memory. Loading
a replacement creates another engine and partition residency before the old one
is released. This overlap is required for non-disruptive handoff and may exceed
device capacity. A load failure rolls back without changing active admission.

`model_memory` reports tensor payload bytes, not allocator, compute-buffer, or
context/KV overhead. The planner does not yet predict temporary overlap bytes.

## Membership and epochs

Discovery is bootstrap and membership input, not scheduling truth. After the
first successful manual deployment establishes intent, the elected coordinator
recomputes the desired plan whenever membership refreshes and on a retry timer.

If `stage_count` is omitted, placement uses all fresh, compatible, memory-
eligible nodes up to the model layer count. An explicit `stage_count` pins the
requested width. A changed node set, API address, eligibility, or layer
assignment produces a new immutable epoch; an equivalent plan is a no-op.

## Safe replacement order

```text
build desired immutable plan
  -> load every replacement partition to ready
  -> activate every replacement partition
  -> publish replacement for new admissions
  -> mark previous runtime partitions draining
  -> wait for previous epoch in-flight count to reach zero
  -> unload previous partitions
```

Direct unload of an active runtime epoch is rejected. `drain` is explicit and
idempotent cleanup makes retries safe after an ambiguous timeout.

## Failure behavior

- Prepare or partial-activation failure drains and unloads the attempted epoch;
  the previous healthy epoch remains active.
- A prepare timeout consumes its epoch and rolls back, preventing delayed work
  from colliding with a later retry.
- Cleanup failure does not unpublish the replacement. The old plan stays visible
  in `draining` status and reconciliation retries it.
- If a runtime is unreachable, reachable old partitions are still cleaned and
  the healthy replacement remains admissible. Exact cleanup is retried when the
  runtime returns.
- If no valid replacement exists and the active route is no longer healthy, the
  deployment becomes `degraded` and rejects new admission.

No live KV migration occurs. A session assigned to a runtime that is lost cannot
be recovered by re-planning; only sessions whose assigned stages remain healthy
can finish.

## Durability boundary

Deployment intent, epochs, and in-flight counters are currently coordinator-
local memory. The local deterministic election is sufficient for current one-
and two-node experiments, but a new coordinator does not reconstruct this state.
Durable replicated deployment state belongs with the future three-voter
consensus milestone.
