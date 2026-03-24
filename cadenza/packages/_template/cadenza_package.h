#pragma once
#ifndef CADENZA_PACKAGE_H
#define CADENZA_PACKAGE_H

#include <cstdint>

#ifndef CADENZA_VERSION
#define CADENZA_VERSION "0.1.0"
#endif

namespace cadenza::packages {

struct PackageMetadata {
    const char* name;
    const char* version;
    uint32_t tick_rate_ms;
    uint8_t preferred_core;
    const char** topics_published;
    uint32_t num_topics_published;
    const char** topics_subscribed;
    uint32_t num_topics_subscribed;
};

class CadenzaPackage {
public:
    virtual ~CadenzaPackage() = default;
    virtual int on_attach() = 0;
    virtual int on_tick() = 0;
    virtual int on_detach() = 0;
    virtual PackageMetadata metadata() const = 0;
};

} // namespace cadenza::packages

// Exported symbol for dlopen-based loading
extern "C" {
    const char* cadenza_abi_version();
    cadenza::packages::CadenzaPackage* cadenza_create_package();
    void cadenza_destroy_package(cadenza::packages::CadenzaPackage* pkg);
}

#endif
