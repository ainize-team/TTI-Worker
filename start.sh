#!/bin/bash

celery -A worker worker --loglevel=info --pool=solo --concurrency=1  --without-heartbeat
