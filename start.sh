#!/bin/bash

# Start Redis in the background
echo "Starting Redis server..."
redis-server --daemonize yes

# Start MongoDB in the background
echo "Starting MongoDB server..."
# Ensure data directory exists
mkdir -p /data/db
mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db --bind_ip 127.0.0.1

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to start..."
until mongosh --eval "db.adminCommand('ping')" --quiet > /dev/null 2>&1; do
  sleep 1
done
echo "MongoDB is ready."

# Start the FastAPI application
echo "Starting FastAPI application..."
exec uv run uvicorn main:app --host 0.0.0.0 --port 8000
