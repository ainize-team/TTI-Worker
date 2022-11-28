#!/bin/bash
. /app/.venv/bin/activate

celery -A worker worker -P threads --concurrency=1 -l INFO --without-heartbeat