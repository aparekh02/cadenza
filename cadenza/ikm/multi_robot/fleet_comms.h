#pragma once
#ifndef CADENZA_IKM_FLEET_COMMS_H
#define CADENZA_IKM_FLEET_COMMS_H

#include <cstdint>
#include <cstddef>

namespace cadenza::ikm {

int fleet_comms_init();
int fleet_comms_discover_peers();
int fleet_comms_broadcast(const void* data, size_t len);
void fleet_comms_shutdown();

} // namespace cadenza::ikm

#endif
