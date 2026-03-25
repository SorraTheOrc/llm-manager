Container image for llama-server
===============================

Overview
--------
This directory contains a Containerfile and build script to produce an image that runs `start-llama.sh` inside a container. The goal is to create a self-contained image that can be run under a system-level container runtime (podman) for production use, avoiding reliance on distrobox and per-user namespaces.

How it works
------------
- `Containerfile` builds a minimal Fedora-based image, copies `start-llama.sh` into `/usr/local/bin` and exposes it as the container entrypoint.
- At runtime the container should be started with the desired `MODEL` argument, and models should be provided via a dedicated host directory (e.g. `/var/lib/llama-models`) mounted read-only into the container.

Build
-----
From the repository root run:

```
cd proxy
./container/build-image.sh
```

This builds the image `llama-proxy/llama-server:local` using podman.

Run (example)
-------------
Run the container with system podman, mounting host model dir and exposing port 8080:

```
podman run -d \
  --name llama-server-local \
  -p 8080:8080 \
  -v /var/lib/llama-models:/models:ro \
  llama-proxy/llama-server:local qwen3
```

Notes
-----
- The Containerfile is minimal and may require additional packages or GPU drivers depending on your environment. Update it to add GPU runtimes / drivers as needed.
- For production, pin the base image and add reproducible builds.
