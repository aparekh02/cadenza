#pragma once
#ifndef CADENZA_IKM_BUS_H
#define CADENZA_IKM_BUS_H

#include <cstddef>
#include <cstdint>
#include <functional>

namespace cadenza::ikm {

static constexpr uint32_t MAX_TOPICS = 256;
static constexpr uint32_t MAX_SUBSCRIBERS_PER_TOPIC = 64;

using SubscriberCallback = std::function<void(uint32_t topic_id, const void* data, size_t len)>;

int ikm_bus_init();
int ikm_publish(uint32_t topic_id, const void* data, size_t len);
int ikm_subscribe(uint32_t topic_id, SubscriberCallback callback);
uint32_t ikm_topic_from_name(const char* name);
void ikm_bus_shutdown();

} // namespace cadenza::ikm

#endif
