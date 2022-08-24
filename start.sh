#!/bin/bash

#. /app/.venv/bin/activate

pip3 uninstall torch -y
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113

celery -A worker worker --loglevel=info --pool=solo
