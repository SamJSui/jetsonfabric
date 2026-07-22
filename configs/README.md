# Configuration

`models.example.json` documents the coordinator's model registry schema and is
sufficient for route-planning examples. Dynamic deployment switching also
requires each model entry to contain the exact local `artifact_path` and
`artifact_sha256` of its GGUF file.

Create an ignored, machine-local registry like this:

```sh
MODEL_PATH="/var/lib/jetsonfabric/models/qwen.gguf"
MODEL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"

jq -n \
  --arg path "$MODEL_PATH" \
  --arg sha256 "$MODEL_SHA256" \
  '{models:[{
    id:"qwen2.5-coder-1.5b-q4",
    family:"llm",
    supported_engines:["llama.cpp"],
    layer_count:28,
    min_memory_gb:3,
    placement_modes:["pipeline_parallel"],
    artifact_path:$path,
    artifact_sha256:$sha256
  }]}' > configs/models.local.json

sudo install -m 0644 configs/models.local.json \
  /etc/jetsonfabric/models.json
```

Confirm that `layer_count` matches the GGUF architecture. For a physical
cluster, put the same model bytes at the same absolute path and install an
equivalent registry on every node so coordinator failover sees the same model
identity. Pass the registry with `--models /etc/jetsonfabric/models.json` or set
`MODELS_PATH=configs/models.local.json` for direct Makefile commands.
