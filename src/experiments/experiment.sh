#!/bin/bash

_term() {
  echo "Caught SIGTERM signal!"
  kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM

echo "Doing some initial work..."
for lr in 0.1 0.01 0.001; do
  for reg in 0.1 0.01 0.001; do
    ./src/main.py --lr $lr --early 8 --datadir datasets/ >> src/experiments/result.txt
  done
done
0.1 ~ 0.5 0.05

/bin/start/main/server --nodaemon &

child=$!
wait "$child"