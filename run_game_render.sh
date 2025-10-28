#!/bin/bash
export PYTHONPATH="$(pwd)/pvz:$(pwd)/gym-pvz:$PYTHONPATH"
python3 game_render.py