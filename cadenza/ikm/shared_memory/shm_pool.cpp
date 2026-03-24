#include "shm_pool.h"
#include <cstring>
#include <atomic>

#ifdef CADENZA_TEST_MODE
#include <cstdlib>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cadenza::ikm {

static constexpr uint32_t MAX_TOPICS_INTERNAL = 256;
static constexpr size_t DEFAULT_TOPIC_SIZE = 256 * 1024; // 256KB per topic

struct TopicSlot {
    uint32_t topic_id;
    size_t offset;
    size_t capacity;
    std::atomic<size_t> used{0};
    bool allocated;
};

static void* shm_base = nullptr;
static size_t shm_total_size = 0;
static TopicSlot topic_slots[MAX_TOPICS_INTERNAL];
static std::atomic<uint32_t> allocated_topics{0};
static size_t next_offset = 0;

#ifdef CADENZA_TEST_MODE
static std::atomic<int64_t> copy_counter{0};

int64_t shm_pool_get_copy_count() {
    return copy_counter.load(std::memory_order_relaxed);
}
#endif

int shm_pool_init(size_t total_size) {
    if (shm_base) return -1; // Already initialized

    shm_total_size = total_size;
    std::memset(topic_slots, 0, sizeof(topic_slots));
    next_offset = 0;
    allocated_topics.store(0, std::memory_order_release);

#ifdef CADENZA_TEST_MODE
    shm_base = std::malloc(total_size);
    if (!shm_base) return -2;
    std::memset(shm_base, 0, total_size);
    copy_counter.store(0, std::memory_order_release);
#else
    int fd = shm_open("/cadenza_ikm", O_CREAT | O_RDWR, 0666);
    if (fd < 0) return -2;

    if (ftruncate(fd, static_cast<off_t>(total_size)) < 0) {
        ::close(fd);
        shm_unlink("/cadenza_ikm");
        return -3;
    }

    shm_base = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);

    if (shm_base == MAP_FAILED) {
        shm_base = nullptr;
        shm_unlink("/cadenza_ikm");
        return -4;
    }
#endif

    return 0;
}

static int ensure_topic_allocated(uint32_t topic_id) {
    // Check if already allocated
    uint32_t count = allocated_topics.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; i++) {
        if (topic_slots[i].topic_id == topic_id && topic_slots[i].allocated)
            return 0;
    }

    // Allocate new slot
    if (count >= MAX_TOPICS_INTERNAL) return -1;
    if (next_offset + DEFAULT_TOPIC_SIZE > shm_total_size) return -2;

    auto& slot = topic_slots[count];
    slot.topic_id = topic_id;
    slot.offset = next_offset;
    slot.capacity = DEFAULT_TOPIC_SIZE;
    slot.used.store(0, std::memory_order_release);
    slot.allocated = true;

    next_offset += DEFAULT_TOPIC_SIZE;
    allocated_topics.store(count + 1, std::memory_order_release);
    return 0;
}

void* shm_pool_get_buffer(uint32_t topic_id) {
    if (!shm_base) return nullptr;
    ensure_topic_allocated(topic_id);

    uint32_t count = allocated_topics.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; i++) {
        if (topic_slots[i].topic_id == topic_id && topic_slots[i].allocated)
            return static_cast<uint8_t*>(shm_base) + topic_slots[i].offset;
    }
    return nullptr;
}

size_t shm_pool_get_capacity(uint32_t topic_id) {
    uint32_t count = allocated_topics.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; i++) {
        if (topic_slots[i].topic_id == topic_id && topic_slots[i].allocated)
            return topic_slots[i].capacity;
    }
    return 0;
}

float shm_pool_utilization() {
    if (shm_total_size == 0) return 0.0f;
    return static_cast<float>(next_offset) / static_cast<float>(shm_total_size);
}

int shm_pool_resize_topic(uint32_t topic_id, size_t new_capacity) {
    uint32_t count = allocated_topics.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count; i++) {
        if (topic_slots[i].topic_id == topic_id && topic_slots[i].allocated) {
            // Can only shrink in-place (no realloc in shared memory)
            if (new_capacity <= topic_slots[i].capacity) {
                topic_slots[i].capacity = new_capacity;
                return 0;
            }
            return -1; // Cannot grow
        }
    }
    return -2; // Topic not found
}

void shm_pool_shutdown() {
    if (!shm_base) return;

#ifdef CADENZA_TEST_MODE
    std::free(shm_base);
#else
    munmap(shm_base, shm_total_size);
    shm_unlink("/cadenza_ikm");
#endif

    shm_base = nullptr;
    shm_total_size = 0;
    next_offset = 0;
    allocated_topics.store(0, std::memory_order_release);
}

} // namespace cadenza::ikm
