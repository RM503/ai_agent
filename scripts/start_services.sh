#!/bin/bash

set -e

SESSION="dev"
PROJECT="../src"

# If session already exists, just attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s $SESSION -n redis "cd $PROJECT && redis-server"
tmux new-window -t $SESSION -n celery "cd $PROJECT && celery -A agent.worker.celery_app:celery_app worker --loglevel=info"
tmux new-window -t $SESSION -n uvicorn "cd $PROJECT && uv run uvicorn agent.main:app --reload"

tmux select-window -t "$SESSION:uvicorn"
exec tmux attach -t "$SESSION"