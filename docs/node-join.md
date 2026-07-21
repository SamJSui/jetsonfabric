# Node Join Flow

Every Jetson runs the same `jetsonfabric-node` process. A node supervises its
local runtime worker, discovers peers, participates in membership, and exposes
the facade API. There are no separate user-facing control and agent services.

## Local Network Join

The default discovery mode is mDNS. On one LAN, a new node can start with a
stable data directory and no coordinator address:

```bash
export JETSONFABRIC_CLUSTER_TOKEN="$(openssl rand -hex 32)"

./dist/jetsonfabric-node-linux-arm64 \
  --node-name dopey \
  --listen 0.0.0.0:52415 \
  --advertise-url http://dopey.local:52415 \
  --data-dir /var/lib/jetsonfabric \
  --runtime-bin /opt/jetsonfabric/jetsonfabric-runtime-worker \
  --runtime-listen 127.0.0.1:9090 \
  --models /etc/jetsonfabric/models.json
```

`--node-name` defaults to a hostname-derived name when omitted. The data
directory stores the stable logical node identity; node names remain operator
labels rather than security identities.

Configure the same `JETSONFABRIC_CLUSTER_TOKEN` value on every node. It is read
from the environment rather than a command-line flag so the secret does not
appear in process arguments. Nodes can still serve static deployments without
it, but runtime load, activate, and unload requests fail closed until it is set.

## Static Bootstrap

When multicast discovery is unavailable, configure one or more reachable node
API URLs:

```bash
./dist/jetsonfabric-node-linux-arm64 \
  --node-name grumpy \
  --discovery static \
  --seeds http://dopey.local:52415
```

Static seeds bootstrap membership. They are not permanent coordinators and do
not become scheduling truth.

## Expected Lifecycle

1. The node binds its facade API and starts or connects to its local runtime.
2. Discovery finds one or more peers.
3. Membership converges on fresh node and runtime capabilities.
4. Eligible coordinator-capable nodes run deterministic election.
5. The elected coordinator creates deployment epochs and routes new sessions.
6. Runtime work remains node-local until a stage request is forwarded to the
   next selected node.

## Security Boundary

P0 assumes a trusted LAN. The shared cluster token authenticates coordinator
lifecycle writes, but HTTP does not encrypt the token or other traffic.
Hostnames, node names, and persisted node IDs are not authentication. Before
exposing a cluster across untrusted networks, add authenticated node enrollment
and mutually authenticated transport; do not expose the current node APIs
directly to the public internet.
