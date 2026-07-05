# mDNS Discovery

JetsonFabric node discovery now supports both static seeds and LAN-local mDNS.

The default discovery mode is:

```text
static,mdns
```

Static discovery still uses configured seed URLs. mDNS discovery broadcasts and browses a local multicast service:

```text
_jetsonfabric._tcp.local.
```

Each node advertises TXT metadata such as:

```text
cluster_id=home-lab
node_id=<stable node id>
node_name=dopey
api_port=52415
control_eligible=true
control_priority=20
```

Peers filter discovered records by `cluster_id`, reconstruct a reachable node API URL from the packet source IP and advertised `api_port`, and merge the resulting member record into their membership table.

## UX goal

For LAN-connected Jetsons, the intended command becomes:

```bash
make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=dopey
```

A second Jetson can run:

```bash
make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=sleepy
```

No `NODE_SEEDS` or raw IPs should be required when mDNS multicast works on the local network.

## Fallbacks

mDNS is LAN-local and may not work across Tailscale-only paths, WSL NAT boundaries, VLANs, or firewalls that block UDP 5353.

When mDNS is unreliable, keep using static discovery with a hostname or Tailscale/MagicDNS name:

```bash
make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=wsl \
  NODE_LISTEN=0.0.0.0:52425 \
  NODE_SEEDS=http://dopey:52415
```

## Discovery modes

Supported modes:

```text
static
mdns
none
```

Examples:

```bash
NODE_DISCOVERY=mdns make node-run
NODE_DISCOVERY=static make node-run NODE_SEEDS=http://dopey:52415
NODE_DISCOVERY=static,mdns make node-run NODE_SEEDS=http://dopey:52415
```
