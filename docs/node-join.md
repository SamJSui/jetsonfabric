# Node Join Flow

Every Jetson runs the same `jetsonfabric-node` process. A node supervises its
local runtime worker, discovers peers, participates in membership, and exposes
the facade API. There are no separate user-facing control and agent services.

## Prepare Each Jetson

Build on each Jetson and create the standard filesystem layout:

```sh
make node-linux-arm64
make runtime-cuda RUNTIME_CUDA_ARCH=87
sh scripts/install-node-layout.sh

sudo install -m 0755 dist/jetsonfabric-node-linux-arm64 \
  /opt/jetsonfabric/jetsonfabric-node
sudo install -m 0755 dist/jetsonfabric-runtime-worker \
  /opt/jetsonfabric/jetsonfabric-runtime-worker
```

Place the same GGUF bytes at the same absolute path on every candidate node.
Then create `/etc/jetsonfabric/models.json` with that path, its SHA-256 digest,
and the model's actual transformer layer count. The exact schema and generation
command are in [`configs/README.md`](../configs/README.md).

Generate the cluster token once and install the same private environment file
on every node:

```sh
TOKEN="$(openssl rand -hex 32)"
printf 'JETSONFABRIC_CLUSTER_TOKEN=%s\n' "$TOKEN" \
  | sudo tee /etc/jetsonfabric/node.env >/dev/null
sudo chmod 0600 /etc/jetsonfabric/node.env
```

Copy the resulting `node.env` to the other nodes over an authenticated channel.
Do not independently generate one token per node.

## Local Network Join

The default discovery mode is mDNS. On one LAN, a new node can start with a
stable data directory and no coordinator address:

```bash
set -a
. /etc/jetsonfabric/node.env
set +a

/opt/jetsonfabric/jetsonfabric-node \
  --node-name dopey \
  --cluster-id home-lab \
  --role jetson \
  --listen 0.0.0.0:52415 \
  --advertise-url http://dopey.local:52415 \
  --data-dir /var/lib/jetsonfabric/node \
  --runtime-bin /opt/jetsonfabric/jetsonfabric-runtime-worker \
  --runtime-listen 127.0.0.1:9090 \
  --runtime-idle \
  --runtime-compute-backend cuda \
  --runtime-mode pipeline_parallel \
  --runtime-n-gpu-layers 999 \
  --runtime-cuda-active \
  --benchmarks /var/lib/jetsonfabric/benchmarks/events.jsonl \
  --models /etc/jetsonfabric/models.json
```

On the second Jetson, source its copy of `node.env`, then change only the node
name, advertised hostname, and local data directory if desired:

```sh
set -a
. /etc/jetsonfabric/node.env
set +a

/opt/jetsonfabric/jetsonfabric-node \
  --node-name grumpy \
  --cluster-id home-lab \
  --role jetson \
  --listen 0.0.0.0:52415 \
  --advertise-url http://grumpy.local:52415 \
  --data-dir /var/lib/jetsonfabric/node \
  --runtime-bin /opt/jetsonfabric/jetsonfabric-runtime-worker \
  --runtime-listen 127.0.0.1:9090 \
  --runtime-idle \
  --runtime-compute-backend cuda \
  --runtime-mode pipeline_parallel \
  --runtime-n-gpu-layers 999 \
  --runtime-cuda-active \
  --benchmarks /var/lib/jetsonfabric/benchmarks/events.jsonl \
  --models /etc/jetsonfabric/models.json
```

`--node-name` defaults to a hostname-derived name when omitted. The data
directory stores the stable logical node identity; node names remain operator
labels rather than security identities.

Configure the same `JETSONFABRIC_CLUSTER_TOKEN` value on every node. It is read
from the environment rather than a command-line flag so the secret does not
appear in process arguments. Coordinator lifecycle writes, runtime generation
gateway calls, and peer Stagewire requests fail closed until it is set.

## Static Bootstrap

When multicast discovery is unavailable, replace the default discovery behavior
in the base command with one or more reachable node API URLs:

```bash
--discovery static \
--seeds http://dopey.local:52415
```

Static seeds bootstrap membership. They are not permanent coordinators and do
not become scheduling truth.

## Verify and Activate

The membership response should contain both `dopey` and `grumpy`, with one
member also present in the `leader` field:

```sh
curl -sS http://dopey.local:52415/v1/cluster/members | jq
```

Activate the registered model as two ordered stages, inspect the immutable
active epoch, and send a completion through either node:

```sh
curl -sS -X POST http://dopey.local:52415/v1/deployments/switch \
  -H 'Content-Type: application/json' \
  --data-binary @examples/deployment-switch-request.json | jq

curl -sS http://dopey.local:52415/v1/deployments/active | jq

curl -sS -X POST http://grumpy.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat-request.json | jq
```

`--runtime-cuda-active` is a node capability attestation required for CUDA
placement. It is not by itself proof of GPU execution; the physical validation
procedure captures the evidence required for that claim.

## Expected Lifecycle

1. The node binds its facade API and starts or connects to its local runtime.
2. Discovery finds one or more peers.
3. Membership converges on fresh node and runtime capabilities.
4. Eligible coordinator-capable nodes run deterministic election.
5. The elected coordinator creates deployment epochs and routes new sessions.
6. Runtime work remains node-local until a stage request is forwarded to the
   next selected node.

## Security Boundary

The current implementation assumes a trusted LAN. The shared cluster token
authenticates coordinator lifecycle and generation calls plus peer Stagewire
requests, but HTTP does not encrypt the token or other traffic.
Hostnames, node names, and persisted node IDs are not authentication. Before
exposing a cluster across untrusted networks, add authenticated node enrollment
and mutually authenticated transport; do not expose the current node APIs
directly to the public internet.
