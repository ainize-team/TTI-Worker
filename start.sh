#!/bin/bash

. /app/.venv/bin/activate
celery -A worker worker --loglevel=info --pool=solo
