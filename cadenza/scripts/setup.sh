#!/bin/bash
# Cadenza OS — First-time setup script
# Run as root on Ubuntu 22.04 LTS (ARM64)
set -e

echo "=== Cadenza OS Setup ==="

# ── Install build dependencies ──
echo "Installing build dependencies..."
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libspdlog-dev \
    libprotobuf-dev \
    libavahi-client-dev \
    libcap-dev

# ── Configure CPU isolation for RT cores ──
echo "Configuring CPU isolation for real-time scheduling..."
GRUB_FILE="/etc/default/grub"
if grep -q "isolcpus" "$GRUB_FILE"; then
    echo "  isolcpus already configured, skipping."
else
    sed -i 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"/' "$GRUB_FILE"
    update-grub
    echo "  Added isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3 to GRUB."
    echo "  REBOOT REQUIRED for CPU isolation to take effect."
fi

# ── Configure RT scheduling limits ──
echo "Configuring real-time scheduling limits..."
LIMITS_FILE="/etc/security/limits.conf"
if grep -q "cadenza" "$LIMITS_FILE"; then
    echo "  RT limits already configured, skipping."
else
    cat >> "$LIMITS_FILE" << 'LIMITS'

# Cadenza OS — RT scheduling limits
@cadenza  soft  rtprio  99
@cadenza  hard  rtprio  99
@cadenza  soft  memlock unlimited
@cadenza  hard  memlock unlimited
@cadenza  soft  nice    -20
@cadenza  hard  nice    -20
LIMITS
    echo "  Added RT limits for @cadenza group."
fi

# ── Create cadenza group if needed ──
if ! getent group cadenza > /dev/null 2>&1; then
    groupadd cadenza
    echo "  Created 'cadenza' group."
fi

# ── Create runtime directories ──
mkdir -p /run/cadenza
mkdir -p /opt/cadenza
mkdir -p ~/.cadenza/models

echo ""
echo "=== Setup Complete ==="
echo ""
echo "First-run instructions:"
echo "  1. Add your user to the cadenza group:  sudo usermod -aG cadenza \$USER"
echo "  2. Reboot to apply CPU isolation:       sudo reboot"
echo "  3. Build Cadenza OS:"
echo "       cd cadenza"
echo "       mkdir build && cd build"
echo "       cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "       make -j\$(nproc)"
echo "  4. Install the CLI:                     sudo make install"
echo "  5. Create your first project:           cadenza init my-robot --robot go1"
echo ""
