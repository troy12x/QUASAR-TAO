FROM rust:1.83-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo.toml and Cargo.lock
COPY Cargo.toml Cargo.lock* ./

# Build dependencies
RUN cargo build --release 2>&1 || true

# Copy source code
COPY src ./src/

# Build the application
RUN cargo build --release

# Environment variables
ENV NETUID=24
ENV WALLET_NAME=validator
ENV HOTKEY=default
ENV CHALLENGE_URL=http://challenge:8080
ENV NETWORK=finney

# Run the validator
CMD ["cargo", "run", "--release"]
