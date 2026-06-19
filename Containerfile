FROM docker.io/rocm/dev-ubuntu-24.04:7.2.4

WORKDIR /opt

# Install build dependencies (package manager varies by base image)
RUN (command -v apt-get && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Options::=\"--force-overwrite\" -o Dpkg::Options::=\"--force-confold\" install -y git build-essential cmake ninja-build ca-certificates libssl-dev libcurl4-openssl-dev pkg-config rocm-dev hipblas rocblas rocwmma-dev7.2.4 && apt-get clean) || \
    (command -v dnf && dnf -y install git cmake make gcc gcc-c++ ninja-build && dnf clean all) || \
    (command -v microdnf && microdnf -y install git cmake make gcc gcc-c++ ninja-build && microdnf clean all) || \
    (echo "No supported package manager found in base image; check base OS" && exit 1)

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /opt/llama.cpp

RUN cmake -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_HIP_ROCWMMA_FATTN=ON \
      -DLLAMA_OPENSSL=ON && \
    cmake --build build --config Release -j"$(nproc)" && \
    ln -sf /opt/llama.cpp/build/bin/* /usr/local/bin/
