#include "transport/media_type.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace runtime = jetsonfabric::runtime;

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_media_type_matching() {
    constexpr const char* expected = "application/vnd.jetsonfabric.stage.v1+octet-stream";
    expect(runtime::transport::matches_media_type(expected, expected), "exact media type was rejected");
    expect(
        runtime::transport::matches_media_type(
            "Application/Vnd.JetsonFabric.Stage.V1+Octet-Stream",
            expected
        ),
        "case-normalized media type was rejected"
    );
    expect(
        runtime::transport::matches_media_type(
            "application/vnd.jetsonfabric.stage.v1+octet-stream; charset=utf-8; version=\"1\"",
            expected
        ),
        "valid media type parameters were rejected"
    );
}

void test_malformed_media_types_are_rejected() {
    constexpr const char* expected = "application/vnd.jetsonfabric.stage.v1+octet-stream";
    const std::string invalid[] = {
        "application/vnd.jetsonfabric.stage.v1+octet-stream-garbage",
        "application/vnd.jetsonfabric.stage.v1+octet-stream;",
        "application/vnd.jetsonfabric.stage.v1+octet-stream; charset",
        "application/vnd.jetsonfabric.stage.v1+octet-stream; charset=",
        "application/vnd.jetsonfabric.stage.v1+octet-stream; charset=\"unterminated",
        "application/vnd.jetsonfabric.stage.v1+octet-stream; =utf-8",
    };
    for (const std::string& value : invalid) {
        expect(!runtime::transport::matches_media_type(value, expected), "malformed media type was accepted: " + value);
    }
}

} // namespace

int main() {
    try {
        test_media_type_matching();
        test_malformed_media_types_are_rejected();
        std::cout << "HTTP media type tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
