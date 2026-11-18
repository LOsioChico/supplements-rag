#!/usr/bin/env bash
# Deploy supplements-rag to Cloudflare Workers.
# 1. Sync the pipeline package into the worker bundle (single source of
#    truth stays in src/supplements_rag — src/supplements_rag under
#    worker/ is generated, never edit there).
# 2. Migrations first — idempotent, and a missing table can never ship
#    silently.
# 3. pywrangler vendors deps + deploys.
# Load the index first when data/ changed:
#   node worker/scripts/load_d1.mjs
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --delete --exclude __pycache__ src/supplements_rag/ worker/src/supplements_rag/
cd worker && npx wrangler d1 migrations apply supplements-rag --remote && uv run pywrangler deploy "$@"
