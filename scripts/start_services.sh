#!/bin/bash

SESSION="dev"
PROJECT="../src"

tmux new-session -d -s $SESSION -n redis "CD $PROJECT && redis-server"
tmux new-window -t $SESSION -n celery "cd $PROJECT && celery -A agent.worker.celery_app:celery_app worker --loglevel=info"
tmux new-window -t $SESSION -n uvicorn "cd $PROJECT && uv run uvicorn agent.main:app --reload"

tmux attach -t $SESSION