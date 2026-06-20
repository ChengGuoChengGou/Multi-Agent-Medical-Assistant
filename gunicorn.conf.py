"""Gunicorn configuration for Multi-Agent Medical Chatbot.
Usage: gunicorn -c gunicorn.conf.py app:app
"""
import multiprocessing
import os

# Server socket
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:5000")
backlog = 2048

# Worker processes
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120  # seconds - medical queries may take longer
keepalive = 5

# Logging
accesslog = os.environ.get("GUNICORN_ACCESS_LOG", "-")
errorlog = os.environ.get("GUNICORN_ERROR_LOG", "-")
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "medical-chatbot"

# Server mechanics
preload_app = True  # Load app before forking (shared memory, faster startup)
max_requests = 1000  # Restart workers after N requests (prevent memory leaks)
max_requests_jitter = 50  # Randomize restart to avoid all workers restarting at once

# SSL (set via env vars if needed)
keyfile = os.environ.get("GUNICORN_KEYFILE")
certfile = os.environ.get("GUNICORN_CERTFILE")
