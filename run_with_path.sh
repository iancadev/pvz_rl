#!/bin/bash
export PYTHONPATH="/tmp/81592/pvz_rl/pvz:/tmp/81592/pvz_rl/gym-pvz:$PYTHONPATH"
cd /tmp/81592/pvz_rl
exec "$@"