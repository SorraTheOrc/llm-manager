FROM docker.io/rocm/dev-ubuntu-24.04:7.2.4

WORKDIR /opt

# Install build dependencies (package manager varies by base image)
RUN if command -v apt-get >/dev/null 2>&1; then \
      export DEBIAN_FRONTEND=noninteractive; \
      apt-get update; \
      # ensure apt-utils present to suppress debconf warnings; tolerate failure
      apt-get install -y --no-install-recommends apt-utils || true; \
      apt-get install -y --no-install-recommends -o Dpkg::Options::="--force-overwrite" -o Dpkg::Options::="--force-confdef" git build-essential cmake ninja-build ca-certificates libssl-dev libcurl4-openssl-dev pkg-config rocm-dev hipblas hipblas-dev hipblas-common-dev hipblaslt hipblaslt-dev rocblas rocblas-dev rocsolver rocsolver-dev rocwmma-dev && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then \
      dnf -y install git cmake make gcc gcc-c++ ninja-build && dnf clean all; \
    elif command -v microdnf >/dev/null 2>&1; then \
      microdnf -y install git cmake make gcc gcc-c++ ninja-build && microdnf clean all; \
    else \
      echo "No supported package manager found in base image; check base OS" && exit 1; \
    fi

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /opt/llama.cpp

RUN cmake -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_HIP_ROCWMMA_FATTN=ON \
      -DLLAMA_OPENSSL=ON \
      -Dhipblas_DIR=/opt/rocm-7.2.4/lib/cmake/hipblas \
      -Drocblas_DIR=/opt/rocm-7.2.4/lib/cmake/rocblas \
      -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.4/lib/cmake:/opt/rocm-7.2.4 && \
    cmake --build build --config Release -j"$(nproc)" && \
    ln -sf /opt/llama.cpp/build/bin/* /usr/local/bin/
