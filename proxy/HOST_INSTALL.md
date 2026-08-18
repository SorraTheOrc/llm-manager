Host install instructions for `llama-server`

This file describes safe ways to install `llama-server` on the host so the proxy's direct fallback (running `start-llama.sh` outside a container) works.

Important: your existing setup uses a container image where `llama-server` is present. Installing on the host trades the containment of a container for a simpler runtime; only proceed if you accept that trade.

1) Quick verification (does the binary already exist somewhere in a container?)

   # list running containers (podman)
   podman ps -a

   # if a container named 'llama' exists, check where the binary lives inside it
   podman exec -it llama which llama-server || true

   If the container contains the binary you can copy it out (podman/docker support `cp`) and place it in `/usr/local/bin`.

   Example:
   # inside container find binary path
   podman exec -it llama which llama-server
   # copy from container to host (adjust container id/name and path)
   podman cp llama:/usr/bin/llama-server /tmp/llama-server
   sudo mv /tmp/llama-server /usr/local/bin/llama-server
   sudo chmod +x /usr/local/bin/llama-server

2) Build from source (generic example using llama.cpp / ggml projects)

   NOTE: This is an example. If your environment uses a different `llama-server` implementation, follow that project's build instructions instead.

   # install build prerequisites (Debian/Ubuntu example)
   sudo apt update
   sudo apt install -y build-essential cmake git libopenblas-dev libomp-dev

   # clone and build
   git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
   cd /tmp/llama.cpp
   make

   # the build produces `main` (or different binary names depending on repo);
   # locate the server binary or build target that matches your `llama-server`.
   # If the project produces `llama-server`, install it to /usr/local/bin
   sudo cp ./bin/llama-server /usr/local/bin/llama-server
   sudo chmod +x /usr/local/bin/llama-server

3) Test the installed binary

   which llama-server
   llama-server --help || true

   # Try starting the server with your normal start script to confirm the proxy fallback will work
   proxy/scripts/start-proxy.sh qwen3

   # Optional: add --verbose to log per-chunk SSE data (STREAM CHUNK lines at
   # INFO level) for debugging stream issues. Default operation logs chunks at
   # DEBUG level only, keeping proxy.log small (LP-0MS9GAN2P002NR4M).
   proxy/scripts/start-proxy.sh qwen3 --verbose

4) Restart the proxy service and verify (if you use a system service unit you created)

   sudo systemctl daemon-reload
   sudo systemctl restart <your-unit-name>
   sudo journalctl -u <your-unit-name> -f

   # Verify proxy health
   curl http://localhost:8000/health

6) Install logrotate drop-in (optional but recommended)

   The proxy uses `TimedRotatingFileHandler` for in-process rotation (every
   6 hours by default) with configurable retention.  A logrotate drop-in
   provides compression and a safety net that survives proxy restarts:

     sudo cp proxy/scripts/llama-proxy-logrotate /etc/logrotate.d/llama-proxy
     sudo logrotate -s /var/lib/logrotate/llama-proxy.status \
       /etc/logrotate.d/llama-proxy

   Configuration knobs (in `config.yaml`, `config-fast.yaml`, or
   `config-cheap.yaml`):

   - `logging.retention_days` — how many days of rotated files to keep.
     Default: **7** (was 90, reduced in LP-0MSNKMXIK004P7TL).
   - `logging.compress_after_days` — plain-text rotated files older than
     this are gzip-compressed to `.gz` (keeps the last day readable without
     decompressing). Set to `0` to disable in-process compression. Default: **1**.
   - `logging.rotation_hours` — rotation interval in hours. Default: **6**.
   - `logging.level` — log level. Default: **INFO**.
   - `logging.verbose_chunks` — write per-chunk SSE data at INFO level.
     Default: **false** (cuts log volume by >99% at default INFO level).
   - `logging.directory` — log directory. Default: **/var/log/llama-proxy**.

   Logrotate settings (in the drop-in):

   - `proxy.log` — rotated daily, kept 7 days, gzip compressed.
   - `llama-server.log` — rotated daily, kept 15 days, gzip compressed.

   The proxy compresses old retained files and prunes files beyond
   `retention_days` on every startup, so the first run after a retention
   change will clean up any backlog.

7) If you prefer not to install on the host

   - Use the container-based fixes (make systemd allow writing to /run/user/$UID/libpod, run service as user, or fix podman rootless setup). Those are safer long-term but require more changes.

Notes and troubleshooting
 - If the build step produces a differently-named binary, update `proxy/scripts/start-proxy.sh` or symlink the built binary to `/usr/local/bin/llama-server`.
 - GPU builds or ROCm builds require additional dependencies (ROCm libs, drivers). Ensure you have the proper hardware drivers before attempting those builds.

If you want, I can produce a small scripted installer tailored to your OS (Debian/Ubuntu or Fedora) that will build and install the binary automatically. Reply with the target OS and I will add it as a commit and a work item.
