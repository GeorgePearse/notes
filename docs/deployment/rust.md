# Rust Hosting Options

Quick comparison of platforms for deploying Rust web services.

## Comparison Table

| Platform | Free Tier | Pricing Model | Cold Starts | Auto-Scale | Docker | Best For |
|----------|-----------|---------------|-------------|------------|--------|----------|
| **Fly.io** | 3 shared VMs | Pay-per-use | No (always-on) | Yes | Yes | Edge/global |
| **Railway** | $5/month credit | Usage-based | No | Yes | Yes | Simple deploys |
| **Shuttle** | 3 projects | Usage-based | No | Yes | No* | Rust-native DX |
| **Coolify** | Self-hosted | Your infra cost | No | Manual | Yes | Full control |
| **Cloud Run** | 2M req/month | Per-request | Yes | Yes | Yes | Serverless |

*Shuttle uses their own build system, not Docker

## Fly.io

Edge deployment platform with global distribution.

### Pros
- Deploy to 30+ regions worldwide
- Built-in Postgres, Redis
- No cold starts (VMs stay running)
- Great CLI (`flyctl`)
- Supports WebSockets, long-running connections

### Cons
- Free tier is limited (3 shared-cpu VMs)
- Can get expensive at scale
- Learning curve for `fly.toml` config

### Pricing
- Free: 3 shared-cpu-1x VMs, 160GB outbound
- Shared CPU: ~$1.94/month per VM
- Dedicated: ~$29/month per VM

### Quick Deploy

```bash
# Install CLI
curl -L https://fly.io/install.sh | sh

# Login and launch
fly auth login
fly launch

# Deploy
fly deploy
```

**fly.toml**:
```toml
app = "my-rust-app"
primary_region = "lhr"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256
```

## Shuttle

Rust-native deployment platform with built-in infrastructure.

### Pros
- **Rust-native** - no Dockerfile needed
- Zero config for common setups
- Built-in Postgres, secrets management
- Extremely simple deploy workflow
- Fast iteration during development

### Cons
- **Platform lock-in** - your code uses Shuttle-specific procedural macros (`#[shuttle_runtime::main]`), making it non-portable to other platforms
- Less control over infrastructure
- Smaller ecosystem than general container platforms
- Limited to Rust only

### Pricing
- Free: 3 projects
- Pro: $20/month (more resources, custom domains)
- Usage-based compute on top

### Quick Deploy

```bash
# Install CLI
cargo install cargo-shuttle

# Login
cargo shuttle login

# Init new project
cargo shuttle init

# Deploy
cargo shuttle deploy
```

**main.rs** (Shuttle-specific):
```rust
use axum::{routing::get, Router};

// Note: This macro ties you to Shuttle's platform
#[shuttle_runtime::main]
async fn main() -> shuttle_axum::ShuttleAxum {
    let router = Router::new().route("/", get(hello_world));
    Ok(router.into())
}

async fn hello_world() -> &'static str {
    "Hello, World!"
}
```

⚠️ **Lock-in Warning**: The `#[shuttle_runtime::main]` macro and `shuttle_*` crates embed platform-specific code into your application. To migrate away, you'd need to refactor your entry point and infrastructure setup. If portability matters, consider Docker-based platforms instead.

## Railway

Simple deploys with excellent developer experience.

### Pros
- GitHub integration (auto-deploy on push)
- Nice dashboard and logs
- Easy environment variables
- Built-in Postgres, Redis, MySQL
- Automatic HTTPS

### Cons
- No free tier (only $5 trial credit)
- Less control over infrastructure
- Limited regions compared to Fly

### Pricing
- Trial: $5 one-time credit
- Pro: $5/month + usage (~$0.000463/min CPU, $0.000231/GB-min RAM)
- Typical small Rust app: $5-15/month

### Quick Deploy

```bash
# Install CLI
npm install -g @railway/cli

# Login and init
railway login
railway init

# Link to project and deploy
railway link
railway up
```

Or just connect GitHub repo in dashboard - it auto-detects Rust and builds.

**railway.toml** (optional):
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "./my-app"
healthcheckPath = "/health"
healthcheckTimeout = 100
```

## Coolify

Self-hosted PaaS - like Heroku but on your own servers.

### Pros
- **Free** (open source, self-hosted)
- Full control over infrastructure
- No vendor lock-in
- Deploy from Git, Docker, or compose
- Built-in SSL via Let's Encrypt

### Cons
- Need to manage your own server(s)
- Less polished than commercial options
- You handle scaling, backups, etc.

### Pricing
- Software: Free (open source)
- You pay: VPS costs (~$5-20/month for small server)
- Cloud version: $5/month (Coolify-managed)

### Quick Deploy

```bash
# On your VPS (Ubuntu/Debian)
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash

# Access dashboard at https://your-server-ip:8000
# Add your Git repo, configure build, deploy
```

**docker-compose.yml** (for Coolify):
```yaml
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=${DATABASE_URL}
```

## Google Cloud Run

Serverless containers - pay only when handling requests.

### Pros
- Generous free tier (2M requests/month)
- Scales to zero (pay nothing when idle)
- Scales up massively
- Integrates with GCP ecosystem
- Supports gRPC

### Cons
- Cold starts (can be 1-5s for Rust)
- Max request timeout: 60 minutes
- More complex setup than others
- GCP learning curve

### Pricing
- Free: 2M requests/month, 360k GB-seconds
- CPU: $0.00002400/vCPU-second
- Memory: $0.00000250/GB-second
- Very cheap for low-traffic apps

### Quick Deploy

```bash
# Install gcloud CLI, authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT

# Build and deploy
gcloud run deploy my-rust-app \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

**cloudbuild.yaml** (optional):
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/my-app', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/my-app']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - 'run'
      - 'deploy'
      - 'my-app'
      - '--image=gcr.io/$PROJECT_ID/my-app'
      - '--region=us-central1'
      - '--platform=managed'
```

## Rust Deployment Tips

### Optimized Dockerfile

Multi-stage build for small images:

```dockerfile
# Build stage
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install SSL certs (needed for HTTPS requests)
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/my-app /usr/local/bin/

EXPOSE 8080
CMD ["my-app"]
```

### Even Smaller with Alpine/musl

```dockerfile
FROM rust:1.75-alpine as builder

RUN apk add --no-cache musl-dev

WORKDIR /app
COPY . .

RUN cargo build --release

# Final image: ~10-20MB
FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /app/target/release/my-app /usr/local/bin/

EXPOSE 8080
CMD ["my-app"]
```

### Release Profile Optimization

In `Cargo.toml`:

```toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit (slower build, faster binary)
panic = "abort"     # Remove unwinding code
strip = true        # Strip symbols
```

### Build Caching

Speed up CI/CD with cargo-chef:

```dockerfile
FROM rust:1.75 as chef
RUN cargo install cargo-chef
WORKDIR /app

FROM chef as planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef as builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/my-app /usr/local/bin/
CMD ["my-app"]
```

## Decision Guide

```
Need global edge deployment?
    └─► Fly.io

Want simplest possible deploys?
    └─► Railway or Shuttle

Want Rust-native DX (no Docker)?
    └─► Shuttle (but accept lock-in)

Want full control + free?
    └─► Coolify (self-hosted)

Want serverless + scale to zero?
    └─► Cloud Run

Low traffic, minimize cost?
    └─► Cloud Run (free tier) or Coolify

High traffic, predictable load?
    └─► Fly.io or Railway

Need maximum portability?
    └─► Any Docker-based option (NOT Shuttle)
```

## Cost Examples (Monthly)

| Scenario | Fly.io | Railway | Shuttle | Cloud Run | Coolify |
|----------|--------|---------|---------|-----------|---------|
| Hobby (1K req/day) | ~$2 | ~$5 | Free | Free | $5 VPS |
| Small (100K req/day) | ~$10 | ~$15 | ~$20 | ~$5 | $10 VPS |
| Medium (1M req/day) | ~$50 | ~$50 | ~$50+ | ~$30 | $20 VPS |

*Estimates vary based on CPU usage, memory, and traffic patterns.*
