# Docker Alternatives

## Container Runtimes (Drop-in Replacements)

### Podman
- Red Hat's daemonless, rootless alternative
- CLI nearly identical to Docker (`alias docker=podman` works)
- No daemon = better security model
- Can run Docker Compose files with `podman-compose` or native `podman compose`
- **Most direct Docker replacement**

### containerd
- The runtime Docker itself uses under the hood
- Can be used standalone with `nerdctl` CLI
- Lighter weight than full Docker

### CRI-O
- Lightweight runtime designed specifically for Kubernetes
- Implements Kubernetes Container Runtime Interface (CRI)

## Different Approaches

### LXC/LXD
- System containers (feel more like VMs)
- Better for running full OS environments
- Long-running services rather than ephemeral containers

### Kata Containers
- Lightweight VMs with container-like experience
- Stronger isolation via hardware virtualization
- Good for multi-tenant environments

### Firecracker
- AWS's microVM technology (powers Lambda/Fargate)
- Sub-second boot times
- Strong isolation with minimal overhead

### gVisor
- Google's user-space kernel
- Runs containers with an extra security boundary
- Intercepts syscalls for additional isolation

## For Development

### Nix/NixOS
- Reproducible builds without containers
- Declarative environment management
- Steep learning curve but powerful

### Devbox
- Nix-based dev environments
- Simpler than raw Nix
- Good DX for local development

### Vagrant
- VM-based development environments
- Heavier than containers
- Good for when you need full OS isolation

## Cloud-Native/Serverless

### Modal
- Abstracts away containers entirely for Python workloads
- Handles scaling, GPU provisioning automatically
- Great for ML/AI workloads

### Fly.io
- Uses Firecracker under the hood
- Deploy containers globally with simple CLI
- Good for edge deployment

## Community Opinions: Podman vs Docker

Sources: [r/selfhosted discussion](https://www.reddit.com/r/selfhosted/comments/1itxtp5/how_many_of_you_use_podman_instead_of_docker/), [Self-Hosted Podcast #93](https://selfhosted.show/93), [XDA](https://www.xda-developers.com/podman-better-docker-self-host/)

### Consistent Opinions (widely agreed)

**Why people switch to Podman:**
- **Docker's licensing/business model** - Docker Hub's paid org requirements and Docker Desktop licensing pushed many to explore alternatives
- **Rootless by default** - Security benefit consistently praised; no daemon running as root
- **Drop-in replacement works** - `alias docker=podman` genuinely works for ~95% of use cases
- **Systemd integration** - Native integration for auto-start on boot, proper service management

**Acknowledged trade-offs:**
- **Rootless has limitations** - Can't bind to ports < 1024, some tools don't work inside containers
- **Slight performance penalty** - Rootless networking tops out at 2-4 Gbps vs Docker's 8-10 Gbps
- **Startup latency** - Docker ~150-200ms vs Podman ~200-300ms (marginal for most use cases)
- **Older distros** - May need manual compilation; `generate systemd` deprecated in favor of quadlets

### The Pragmatic Take

Most switchers moved for **business/licensing reasons** rather than pure technical superiority. For self-hosters doing nothing "too advanced," Podman is effectively equivalent. The security benefits (rootless, daemonless) are real but matter most in multi-tenant or high-security environments.

Docker still leads with 59% developer adoption (Stack Overflow 2024), but the gap is closing.

## Summary

| Tool | Best For | Docker Compatibility |
|------|----------|---------------------|
| Podman | Direct Docker replacement | High (same CLI) |
| containerd + nerdctl | Kubernetes environments | Medium |
| LXC/LXD | System containers / VMs | Low |
| Kata Containers | High-security multi-tenant | Medium |
| Firecracker | Serverless / microVMs | Low |
| Nix | Reproducible dev environments | None |
| Modal | Python/ML workloads | None |
