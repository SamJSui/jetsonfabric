#include "protocol/kv_cache_type.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    using namespace jetsonfabric::runtime;

    expect(parse_kv_cache_type("f16") == KVCacheType::F16, "f16 did not parse");
    expect(parse_kv_cache_type("q8_0") == KVCacheType::Q8_0, "q8_0 did not parse");
    expect(kv_cache_type_string(KVCacheType::F16) == "f16", "f16 did not serialize");
    expect(kv_cache_type_string(KVCacheType::Q8_0) == "q8_0", "q8_0 did not serialize");

    bool rejected = false;
    try {
        (void) parse_kv_cache_type("q2_k");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "unsupported KV cache type was accepted");

    std::cout << "KV cache type tests passed\n";
    return 0;
}
