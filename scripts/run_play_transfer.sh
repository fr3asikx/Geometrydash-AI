#!/usr/bin/env bash
set -euo pipefail
python -m geometrydash_ai.ppo.play_transfer --model models/ppo_gd_level1.zip --fps 240
