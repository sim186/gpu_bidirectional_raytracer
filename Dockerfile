FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# freeglut3-dev: for GLUT
# libglew-dev: for GLEW
# cmake: for build system
# clang-format, clang-tidy: for code quality
# git, build-essential: standard tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    freeglut3-dev \
    libglew-dev \
    libglu1-mesa-dev \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    clang-format \
    clang-tidy \
    doxygen \
    graphviz \
    qt6-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
