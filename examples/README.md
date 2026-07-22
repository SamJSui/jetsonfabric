# Request Examples

These files are executable request bodies for the public node API. Run the
commands from the repository root and replace the node URL when necessary.

Activate a registered model as a two-stage deployment:

```sh
curl -sS -X POST http://dopey.local:52415/v1/deployments/switch \
  -H 'Content-Type: application/json' \
  --data-binary @examples/deployment-switch-request.json | jq
```

Send a buffered chat completion through any cluster node:

```sh
curl -sS -X POST http://grumpy.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat-request.json | jq
```

The chat API currently supports `model`, `messages`, `max_tokens`,
`max_completion_tokens`, `stream`, and the optional `jetsonfabric` routing
object. Unsupported OpenAI fields are not included in the tracked example.
