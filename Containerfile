FROM docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7rc-rocwmma

WORKDIR /opt

# Install build dependencies (package manager varies by base image)
RUN (command -v dnf && dnf -y install git cmake make gcc gcc-c++ ninja-build && dnf clean all) || \
    (command -v microdnf && microdnf -y install git cmake make gcc gcc-c++ ninja-build && microdnf clean all) || \
    (echo "No dnf/microdnf found in base image; check base OS" && exit 1)

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /opt/llama.cpp

RUN cmake -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_HIP_ROCWMMA_FATTN=ON && \
    cmake --build build --config Release -j"$(nproc)" && \
    ln -sf /opt/llama.cpp/build/bin/* /usr/local/bin/