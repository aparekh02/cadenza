cmake_minimum_required(VERSION 3.20)
project(CadenzaOS VERSION 0.1.0 LANGUAGES CXX)

# ── ARM64 target check ──────────────────────────────────────────────────────
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    if(NOT CADENZA_TEST_MODE)
        message(FATAL_ERROR
            "Cadenza OS targets ARM64 only. Detected: ${CMAKE_SYSTEM_PROCESSOR}\n"
            "Set -DCADENZA_TEST_MODE=1 to build for testing on non-ARM64 hosts.")
    else()
        message(WARNING "Non-ARM64 host detected (${CMAKE_SYSTEM_PROCESSOR}). "
                        "Building in TEST_MODE — ARM intrinsics disabled.")
    endif()
endif()

# ── C++20 ────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compile-time version define ──────────────────────────────────────────────
add_compile_definitions(CADENZA_VERSION="${PROJECT_VERSION}")

# ── Options ──────────────────────────────────────────────────────────────────
option(CADENZA_TEST_MODE "Build for testing on non-ARM64 hosts" OFF)

if(CADENZA_TEST_MODE)
    add_compile_definitions(CADENZA_TEST_MODE=1)
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Include paths ────────────────────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ── Libraries ────────────────────────────────────────────────────────────────

# IKM — Inter-Kernel Middleware (pub/sub bus)
add_library(cadenza_ikm STATIC
    ikm/shared_memory/shm_pool.cpp
    ikm/ipc/ikm_bus.cpp
    ikm/multi_robot/fleet_comms.cpp
)
target_link_libraries(cadenza_ikm PUBLIC Threads::Threads)
if(NOT CADENZA_TEST_MODE)
    target_link_libraries(cadenza_ikm PUBLIC rt)
endif()

# DSK — Deterministic Scheduling Kernel
add_library(cadenza_dsk STATIC
    dsk/scheduler/edf_scheduler.cpp
    dsk/scheduler/rms_scheduler.cpp
    dsk/scheduler/priority_queue.cpp
    dsk/timer/hpet_timer.cpp
    dsk/watchdog/task_watchdog.cpp
)
target_link_libraries(cadenza_dsk PUBLIC cadenza_ikm Threads::Threads)
if(NOT CADENZA_TEST_MODE)
    target_link_libraries(cadenza_dsk PUBLIC rt)
endif()

# Package System
add_library(cadenza_packages STATIC
    packages/package_loader.cpp
    packages/package_registry.cpp
)
target_link_libraries(cadenza_packages PUBLIC cadenza_ikm ${CMAKE_DL_LIBS})

# Example packages as shared libs
add_library(example_sensor_package SHARED
    packages/_template/example_sensor_package.cpp
)
target_link_libraries(example_sensor_package PRIVATE cadenza_ikm)

add_library(example_sim_package SHARED
    packages/_template/example_sim_package.cpp
)
target_link_libraries(example_sim_package PRIVATE cadenza_ikm)

# AICore — Multimodal intelligence kernel
add_library(cadenza_aicore STATIC
    aicore/perception_fusion.cpp
    aicore/reasoning_engine.cpp
    aicore/aicore_main.cpp
)
target_link_libraries(cadenza_aicore PUBLIC cadenza_ikm cadenza_dsk Threads::Threads)

# ── CLI Binary ───────────────────────────────────────────────────────────────
add_executable(cadenza_cli
    cli/main.cpp
    cli/commands/cmd_init.cpp
    cli/commands/cmd_attach.cpp
    cli/commands/cmd_detach.cpp
    cli/commands/cmd_build.cpp
    cli/commands/cmd_deploy.cpp
    cli/commands/cmd_shell.cpp
    cli/commands/cmd_kernel.cpp
    cli/commands/cmd_rl.cpp
    cli/commands/cmd_sim.cpp
    cli/commands/cmd_log.cpp
    cli/commands/cmd_aios.cpp
    cli/commands/cmd_ai.cpp
)
set_target_properties(cadenza_cli PROPERTIES OUTPUT_NAME cadenza)
target_link_libraries(cadenza_cli PRIVATE
    cadenza_aicore cadenza_dsk cadenza_packages
    Threads::Threads
)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

set(CADENZA_TEST_SOURCES
    tests/test_dsk_scheduling.cpp
    tests/test_ikm_throughput.cpp
    tests/test_package_hotload.cpp
)

foreach(test_src ${CADENZA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE
        cadenza_dsk cadenza_packages
        Threads::Threads
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS cadenza_cli DESTINATION bin)
install(DIRECTORY hal/transducers/ DESTINATION include/cadenza/hal/transducers FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ikm/ DESTINATION include/cadenza/ikm FILES_MATCHING PATTERN "*.h")
install(DIRECTORY dsk/scheduler/ DESTINATION include/cadenza/dsk/scheduler FILES_MATCHING PATTERN "*.h")
install(DIRECTORY kernel_sdk/ DESTINATION include/cadenza/kernel_sdk FILES_MATCHING PATTERN "*.h")
install(FILES packages/_template/cadenza_package.h DESTINATION include/cadenza/packages)
