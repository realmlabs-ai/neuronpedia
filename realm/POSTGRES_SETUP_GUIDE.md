# PostgreSQL Setup for Local Neuronpedia

This guide walks you through setting up PostgreSQL for running Neuronpedia locally.

## Option 1: Install PostgreSQL on Ubuntu (Recommended for Ubuntu)

### Install PostgreSQL
```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Switch to postgres user and set up database
# First try to set password for existing postgres user, if that fails create the user
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" || sudo -u postgres psql -c "CREATE USER postgres WITH SUPERUSER PASSWORD 'postgres';"

# Create postgres database if it doesn't exist
sudo -u postgres createdb postgres || echo "Database postgres already exists"
```

### Verify Installation
```bash
# Test connection
psql -h localhost -U postgres -d postgres

# In psql prompt, you should see:
# postgres=#

# Exit with \q
\q
```

## Option 2: Use Docker (Cross-platform)

### Run PostgreSQL Container
```bash
# Start PostgreSQL in Docker
docker run --name neuronpedia-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  -d postgres:16

# Verify it's running
docker ps
```

### Stop/Start Container
```bash
# Stop
docker stop neuronpedia-postgres

# Start again
docker start neuronpedia-postgres

# Remove completely (will lose data)
docker rm -f neuronpedia-postgres
```

## Option 3: PostgreSQL APT Repository (Latest Version)

```bash
# Add official PostgreSQL APT repository
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list

# Update and install latest PostgreSQL
sudo apt update
sudo apt install postgresql-16 postgresql-contrib-16
```

## Setup Neuronpedia Database

### 1. Install pgvector Extension (Required)

Neuronpedia requires the pgvector extension for vector operations. Install it before running migrations:

```bash
# Install build dependencies
sudo apt update
sudo apt install build-essential postgresql-server-dev-16

# Clone and build pgvector
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable pgvector extension in your database
psql -h localhost -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

For other systems:
- **Docker**: The postgres:16 image already includes pgvector
- **macOS with Homebrew**: `brew install pgvector`

### 2. Install Dependencies
```bash
cd apps/webapp
npm install
```

### 3. Run Database Migrations
```bash
# Create and apply database schema
npm run migrate:localhost
```

This will:
- Create all necessary tables (Model, Source, Feature, Explanation, etc.)
- Set up indexes and constraints
- Prepare the database for Neuronpedia

### 3. (Optional) Seed with Sample Data
```bash
npm run db:seed
```

## Verify Database Setup

### Check Tables
```bash
# Connect to database
psql postgres://postgres:postgres@localhost:5432/postgres

# List tables
\dt

# You should see tables like:
# Model, Source, SourceSet, Feature, Explanation, Activation, etc.

# Exit
\q
```

### Test Web Connection
```bash
cd apps/webapp
npm run dev:localhost
```

Visit http://localhost:3000 - you should see the Neuronpedia interface.

## Connection Details

The webapp uses these default connection settings (from `.env.localhost`):

```
Host: localhost
Port: 5432
Database: postgres
Username: postgres
Password: postgres
```

## Troubleshooting

### "Connection refused" Error
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql
# or for Docker:
docker ps | grep postgres

# Start if not running
sudo systemctl start postgresql
# or for Docker:
docker start neuronpedia-postgres
```

### "Database does not exist" Error
```bash
# Create the database
createdb postgres

# Or with explicit user
createdb -U postgres postgres
```

### "Role does not exist" Error
```bash
# Create the postgres user with superuser privileges
sudo -u postgres createuser -s postgres

# Set password for postgres user
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
```

### Permission Errors
```bash
# Reset PostgreSQL permissions (Ubuntu)
sudo systemctl restart postgresql

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# For Docker, recreate container with correct permissions
docker rm -f neuronpedia-postgres
# Then run the docker command from Option 2 again
```

### Migration Errors
```bash
# Reset and rerun migrations
cd apps/webapp
rm -rf prisma/migrations
npm run migrate:localhost
```

## Next Steps

Once PostgreSQL is running and migrations are complete:

1. **Start the webapp**: `npm run dev:localhost`
2. **Upload your explanations**: `python upload_explanations_local.py /path/to/explanations`
3. **View in browser**: http://localhost:3000

## Advanced: Custom Database

If you want to use a different database name or credentials:

1. **Create your database**:
   ```bash
   createdb myapp
   createuser -P myuser
   ```

2. **Update `.env.localhost`**:
   ```
   POSTGRES_PRISMA_URL="postgres://myuser:mypass@localhost:5432/myapp?pgbouncer=true&connect_timeout=15"
   POSTGRES_URL_NON_POOLING="postgres://myuser:mypass@localhost:5432/myapp"
   ```

3. **Run migrations**:
   ```bash
   npm run migrate:localhost
   ```