# VidyaAI Docker Setup

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- `.env` file with required API keys

### Running with Docker Compose

1. **Build and start all services:**
   ```bash
   docker-compose up --build -d
   ```

2. **Check service status:**
   ```bash
   docker-compose ps
   ```

3. **View logs:**
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f app
   docker-compose logs -f redis
   docker-compose logs -f mongodb
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

5. **Stop services (data persists):**
   ```bash
   docker-compose down
   ```

6. **Stop and remove volumes (clean slate):**
   ```bash
   docker-compose down -v
   ```

## Services

### Application (app)
- **Port:** 8000
- **Depends on:** Redis, MongoDB
- **Health check:** `/health` endpoint

### Redis
- **Port:** 6379
- **Volume:** `redis_data` (persists cache data)
- **Configuration:** AOF persistence enabled

### MongoDB
- **Port:** 27017
- **Volume:** `mongodb_data` (persists database)

## Data Persistence

Both Redis and MongoDB use named volumes to persist data:
- `redis_data`: Stores Redis cache and active sessions
- `mongodb_data`: Stores MongoDB chat history and sessions

**Data survives container restarts** unless you explicitly remove volumes with `docker-compose down -v`.

## Environment Configuration

### For Docker (default):
```env
MONGODB_URI=mongodb://mongodb:27017
REDIS_URL=redis://redis:6379
```

### For Local Development:
```env
MONGODB_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
```

See `.env.example` for all configuration options.

## Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart
```

### Port conflicts
If ports 8000, 6379, or 27017 are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

### Reset everything
```bash
# Stop and remove all containers, networks, and volumes
docker-compose down -v

# Rebuild from scratch
docker-compose up --build -d
```

## Development Workflow

### Rebuild after code changes:
```bash
docker-compose up --build app
```

### Access MongoDB shell:
```bash
docker-compose exec mongodb mongosh
```

### Access Redis CLI:
```bash
docker-compose exec redis redis-cli
```

### View container resource usage:
```bash
docker stats
```
