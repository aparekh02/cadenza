#pragma once
#ifndef CADENZA_HAL_TRANSDUCER_API_H
#define CADENZA_HAL_TRANSDUCER_API_H

#include <cstdint>
#include <cstddef>

namespace cadenza::hal {

enum class TransducerID : uint16_t {
    IMU_PRIMARY = 0,
    IMU_SECONDARY,
    TOF_FRONT,
    TOF_REAR,
    LIDAR_MAIN,
    MOTOR_FL_HIP, MOTOR_FL_THIGH, MOTOR_FL_KNEE,
    MOTOR_FR_HIP, MOTOR_FR_THIGH, MOTOR_FR_KNEE,
    MOTOR_RL_HIP, MOTOR_RL_THIGH, MOTOR_RL_KNEE,
    MOTOR_RR_HIP, MOTOR_RR_THIGH, MOTOR_RR_KNEE,
    MAX_TRANSDUCERS = 256
};

struct TransducerData {
    uint64_t timestamp_us;
    uint16_t transducer_id;
    uint16_t data_len;
    uint8_t data[256];
};

int transducer_read(TransducerID id, TransducerData* buf);
int transducer_write(TransducerID id, const TransducerData* cmd);
int hak_init();

} // namespace cadenza::hal

#endif
