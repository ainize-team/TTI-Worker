#!/bin/bash
celery -A worker worker -P threads --concurrency=1 -l INFO --without-heartbeat