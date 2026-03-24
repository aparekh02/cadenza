#pragma once
#ifndef CADENZA_IKM_SHM_POOL_H
#define CADENZA_IKM_SHM_POOL_H

#include <cstddef>
#include <cstdint>

namespace cadenza::ikm {

struct TopicBuffer {
    uint32_t topic_id;
    size_t offset;
    size_t capacity;
};

int shm_pool_init(size_t total_size = 64 * 1024 * 1024);
void* shm_pool_get_buffer(uint32_t topic_id);
size_t shm_pool_get_capacity(uint32_t topic_id);
float shm_pool_utilization();
int shm_pool_resize_topic(uint32_t topic_id, size_t new_capacity);
void shm_pool_shutdown();

} // namespace cadenza::ikm

#endif
