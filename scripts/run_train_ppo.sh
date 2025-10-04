#!/usr/bin/env bash
set -euo pipefail
python -m geometrydash_ai.ppo.train_ppo --level levels/level1.csv --timesteps 2000000
