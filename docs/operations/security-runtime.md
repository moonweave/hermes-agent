# Investment and tech-feed runtime security update

The two deployed Telegram profiles retain their existing application commits.
This update changes dependency pins and the corresponding lazy-install specs,
without importing the newer engine history or changing profile configuration.

`requirements/security-runtime.in` records the selected runtime packages.
Regenerate its hashed lock with:

```sh
uv pip compile requirements/security-runtime.in --python-version 3.11 \
  --generate-hashes --no-emit-index-url -o requirements/security-runtime.lock
```

Build a separate candidate environment, never the running environment:

```sh
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements/security-runtime.lock
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
```

Run the canonical test wrapper for vision, MCP, access policy, profile isolation,
and profile-specific extensions. Query the installed package versions against
OSV again before switching either service. Preserve the previous release and
service definition for rollback; switch one profile at a time after draining.

The universal project lock includes optional OTLP and Mistral SDK combinations
that require the older OpenTelemetry SDK family. These extras are not installed
in the two selected Telegram runtimes. Their separately compiled runtime lock
uses patched `opentelemetry-api==1.45.2`; it is not a replacement lock for all
provider extras. Use this deployment lock when reproducing these two profiles.

The feed producer selects its dedicated executable through the existing
`TECH_WIKI_HERMES_BIN` setting. Do not repair its delivery by rewriting the
shared `hermes` launcher used by other automation. Transport discovery can be
verified with `hermes send --list telegram --json` without sending a message.

No production conversation or investment job is replayed during verification.
