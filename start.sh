#!/bin/bash

. /app/.venv/bin/activate
celery -A tti_worker worker --loglevel=info --pool=solo
