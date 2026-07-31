#pragma once

#include <stdexcept>
#include <string>

namespace jetsonfabric::runtime {

enum class KVCacheType {
    F16,
    Q8_0,
};

inline KVCacheType parse_kv_cache_type(const std::string& value) {
    if (value == "f16") {
        return KVCacheType::F16;
    }
    if (value == "q8_0") {
        return KVCacheType::Q8_0;
    }
    throw std::invalid_argument("unknown KV cache type: " + value);
}

inline std::string kv_cache_type_string(KVCacheType type) {
    switch (type) {
    case KVCacheType::F16:
        return "f16";
    case KVCacheType::Q8_0:
        return "q8_0";
    }
    return "f16";
}

} // namespace jetsonfabric::runtime
