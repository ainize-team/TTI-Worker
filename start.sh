#!/bin/bash

celery -A worker worker --loglevel=info --pool=solo
