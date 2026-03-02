from celery import Celery

celery_app = Celery(
    "ai_agent",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

# Configurations
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60
)