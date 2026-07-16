# Distributed Inference State Machine

This document defines the engine-neutral lifecycle for one JetsonFabric inference
session. It is an executable design contract backed by `internal/inference` unit
tests. It does not claim that partial-layer execution, activation transport, or
persistent distributed KV cache is implemented yet.

## Scope

The state machine defines:

- coordinator and runtime ownership;
- prefill and decode phases;
- legal session transitions;
- count-based stage positions;
- legal semantic payload transitions;
- ordering, failure, cancellation, and expiration rules.

Transport framing, tensor layout, engine APIs, and scheduling policy are separate
concerns.

## Ownership

### Coordinator

The elected coordinator owns the request-level lifecycle:

- generate `session_id` and `request_id`;
- freeze one model and ordered stage plan for the session;
- initiate prefill;
- receive each sampled token from the final stage;
- decide whether to finish or advance the decode loop;
- increment `decode_step`;
- propagate cancellation and failure;
- request cleanup from every stage;
- assemble the user-facing response.

### First runtime stage

During prefill, the first stage accepts external text or pre-tokenized input,
performs tokenization when needed, executes its assigned transformer layers, and
emits an activation unless it is also the final stage.

During decode, the first stage accepts the previously sampled token, executes its
assigned layers using its local session state, and emits an activation unless it
is also the final stage.

### Intermediate runtime stages

An intermediate stage accepts an activation, executes its assigned layer range,
and emits the next activation during both prefill and decode.

### Final runtime stage

The final stage accepts the preceding activation, executes its assigned layers,
keeps logits engine-local, samples the next token, and returns a `sampled_token`.
For `stage_count=1`, the same runtime is both first and final.

### Every runtime stage

Each runtime stage eventually owns session-local engine state for its assigned
layers, including its local KV cache. A session is bound to one model, one stage
index/count, and one layer range for its entire lifetime.

## Phases

JetsonFabric defines two computational phases:

- `prefill`: process the initial prompt sequence and produce the first sampled
  token;
- `decode`: process one previously sampled token and produce the next sampled
  token.

Finishing, cancellation, failure, and expiration are lifecycle states rather
than model-computation phases.

## Session states

```text
created
   |
   | start_prefill
   v
prefilling
   |\
   | \ begin_finish
   |  v
   | finishing --complete--> completed
   |
   | begin_decode
   v
decoding --advance_decode--> decoding
   |
   | begin_finish
   v
finishing --complete--> completed
```

Active states may also transition to:

```text
cancelled
failed
expired
```

The terminal states are:

```text
completed
cancelled
failed
expired
```

Terminal states reject additional lifecycle events.

## Legal transitions

| Current state | Event | Next state |
|---|---|---|
| `created` | `start_prefill` | `prefilling` |
| `prefilling` | `begin_decode` | `decoding` |
| `prefilling` | `begin_finish` | `finishing` |
| `decoding` | `advance_decode` | `decoding` |
| `decoding` | `begin_finish` | `finishing` |
| `finishing` | `complete` | `completed` |
| any active state | `cancel` | `cancelled` |
| any active state | `fail` | `failed` |
| any active state | `expire` | `expired` |

Examples of rejected transitions include:

- `created -> advance_decode`;
- `prefilling -> complete`;
- `completed -> start_prefill`;
- any event after cancellation, failure, or expiration.

## Stage position

Stage position is represented only by:

```text
stage_index
stage_count
```

A valid position satisfies:

```text
stage_count > 0
0 <= stage_index < stage_count
```

Derived properties are:

```text
is_first = stage_index == 0
is_last  = stage_index == stage_count - 1
```

For `stage_count=1`, index `0` is both first and last. There is no stage-role
enum or wire string.

## Payload transitions

### Prefill

| Stage position | Allowed input | Required output |
|---|---|---|
| first, not final | `text` or `tokens` | `activation` |
| intermediate | `activation` | `activation` |
| final, not first | `activation` | `sampled_token` |
| one stage | `text` or `tokens` | `sampled_token` |

The normal client path starts with `text`. `tokens` permits a future
pre-tokenized caller without changing the downstream stage contract.

### Decode

| Stage position | Required input | Required output |
|---|---|---|
| first, not final | `sampled_token` | `activation` |
| intermediate | `activation` | `activation` |
| final, not first | `activation` | `sampled_token` |
| one stage | `sampled_token` | `sampled_token` |

Logits and KV cache are never inter-stage payloads. They remain internal to the
engine that owns the relevant layers.

## Decode-step ordering

The target protocol uses:

```text
decode_step = 0     prefill
decode_step = 1..N  autoregressive decode steps
```

The coordinator advances the step monotonically. The initial implementation
should permit only one in-flight operation per session. A runtime must reject a
request that changes the model, stage assignment, or layer range of an existing
session.

Retry semantics remain a protocol-design decision: a duplicate step must either
be idempotently replayed or explicitly rejected. Silent re-execution is not an
acceptable default.

## Finish conditions

The coordinator begins finishing when any of the following is true:

- the final stage samples an end-of-generation token;
- the request reaches `max_tokens`;
- the client cancels the request;
- a stage fails;
- a session expires.

Normal completion asks every runtime to release session-local resources before
the coordinator reports `completed`. Failure and cancellation are whole-session
outcomes; JetsonFabric does not continue decoding on a partial stage plan.

## Current implementation status

The current production path still performs one text-carrying pass through the
ordered stage list:

```text
text -> full-model stage generation -> text -> full-model stage generation
```

`internal/inference` defines the target lifecycle and validates its invariants,
but `stageexec`, the Go/C++ wire protocol, and the C++ engine adapter do not yet
run this state machine.

## Deferred work

The following belong to later roadmap steps:

- add phase and lifecycle operations to the Go/C++ wire contract;
- determine llama.cpp partial-layer feasibility;
- define engine-neutral typed stage inputs and outputs;
- implement binary activation framing;
- add persistent runtime sessions and per-stage KV-cache ownership;
- compare distributed greedy output with a single-runtime baseline;
- validate CUDA execution on physical Jetsons.
