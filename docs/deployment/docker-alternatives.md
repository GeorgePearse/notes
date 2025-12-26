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
