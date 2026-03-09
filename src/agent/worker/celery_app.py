# Celery app for distributed tasks
# Run from package root as celery -A agent.worker.celery_app:celery_app worker --loglevel=info
import os

from celery import Celery

celery_app = Celery(
    "ai_agent",
    broker=os.getenv("REDIS_BROKER"), # stores task queues
    backend=os.getenv("REDIS_BACKEND") # stores task results
)

# Auto-discover tasks from 'agent/worker' directory
# No need to hard-code them
celery_app.autodiscover_tasks(["agent.worker"])

# Configurations
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60
)