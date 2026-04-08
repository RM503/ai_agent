#!/bin/bash

set -e

SESSION="dev"
BACKEND="../src"
FRONTEND="../frontend"

# If session already exists, just attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s $SESSION -n redis "cd $BACKEND && redis-server"
tmux new-window -t $SESSION -n celery "cd $BACKEND && celery -A agent.worker.celery_app:celery_app worker --loglevel=info"
tmux new-window -t $SESSION -n uvicorn "cd $BACKED && uv run uvicorn agent.main:app --reload"
tmux new-window -t $SESSION -n "cd $FRONTEND && streamlit run app.py"

tmux select-window -t "$SESSION:uvicorn"
exec tmux attach -t "$SESSION"