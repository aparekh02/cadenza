#include "fleet_comms.h"
#include "../ipc/ikm_bus.h"
#include <cstring>
#include <atomic>

#ifdef CADENZA_TEST_MODE
// Mock fleet comms for testing
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace cadenza::ikm {

static constexpr int MAX_PEERS = 16;
static constexpr uint16_t MULTICAST_PORT = 5353;

struct PeerInfo {
    char hostname[64];
    uint32_t ip_addr;
    bool active;
};

static PeerInfo peers[MAX_PEERS];
static int peer_count = 0;
static int multicast_fd = -1;
static std::atomic<bool> comms_initialized{false};

int fleet_comms_init() {
    if (comms_initialized.load(std::memory_order_acquire)) return -1;

    std::memset(peers, 0, sizeof(peers));
    peer_count = 0;

#ifdef CADENZA_TEST_MODE
    multicast_fd = 500; // Fake FD
#else
    multicast_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (multicast_fd < 0) return -2;

    int reuse = 1;
    ::setsockopt(multicast_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(MULTICAST_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(multicast_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(multicast_fd);
        multicast_fd = -1;
        return -3;
    }

    // Join multicast group 239.0.0.1
    struct ip_mreq mreq{};
    mreq.imr_multiaddr.s_addr = inet_addr("239.0.0.1");
    mreq.imr_interface.s_addr = INADDR_ANY;
    ::setsockopt(multicast_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
#endif

    comms_initialized.store(true, std::memory_order_release);
    return 0;
}

int fleet_comms_discover_peers() {
    if (!comms_initialized.load(std::memory_order_acquire)) return -1;

#ifdef CADENZA_TEST_MODE
    // Add a mock peer
    std::strncpy(peers[0].hostname, "cadenza-test-peer", sizeof(peers[0].hostname) - 1);
    peers[0].ip_addr = 0x7F000002; // 127.0.0.2
    peers[0].active = true;
    peer_count = 1;
    return 1;
#else
    // mDNS-SD discovery would go here (Avahi)
    // For now, return current peer count
    return peer_count;
#endif
}

int fleet_comms_broadcast(const void* data, size_t len) {
    if (!comms_initialized.load(std::memory_order_acquire)) return -1;
    if (!data || len == 0) return -2;

#ifdef CADENZA_TEST_MODE
    return 0;
#else
    struct sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(MULTICAST_PORT);
    dest.sin_addr.s_addr = inet_addr("239.0.0.1");

    ssize_t sent = ::sendto(multicast_fd, data, len, 0,
                            reinterpret_cast<struct sockaddr*>(&dest), sizeof(dest));
    return (sent == static_cast<ssize_t>(len)) ? 0 : -3;
#endif
}

void fleet_comms_shutdown() {
#ifndef CADENZA_TEST_MODE
    if (multicast_fd >= 0) {
        ::close(multicast_fd);
        multicast_fd = -1;
    }
#endif
    comms_initialized.store(false, std::memory_order_release);
    peer_count = 0;
}

} // namespace cadenza::ikm
