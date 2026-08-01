#include "activation/activation_codec_factory.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime = jetsonfabric::runtime;

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void expect(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void append_float(std::vector<std::uint8_t>& bytes, float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bytes.push_back(static_cast<std::uint8_t>(bits & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>((bits >> 16U) & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>((bits >> 24U) & 0xffU));
}

float read_float(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
    return std::bit_cast<float>(bits);
}

runtime::inference::Payload f32_activation() {
    std::vector<std::uint8_t> bytes;
    for (const float value : {0.0F, -0.0F, 1.0F, -2.0F, 0.3333F, 65504.0F}) {
        append_float(bytes, value);
    }
    return runtime::inference::Payload{
        .kind = runtime::inference::PayloadKind::Activation,
        .encoding = "",
        .tensor = runtime::inference::TensorDescriptor{
            .dtype = "f32",
            .shape = {1, 6},
            .byte_order = "little",
            .layout = "row_major",
        },
        .bytes = std::move(bytes),
    };
}

class MisnamedCodec final : public runtime::activation::ActivationCodec {
public:
    const std::string& name() const noexcept override {
        static const std::string value = "actual";
        return value;
    }

    runtime::inference::Payload encode(runtime::inference::Payload payload) const override {
        return payload;
    }

    runtime::inference::Payload decode(runtime::inference::Payload payload) const override {
        return payload;
    }
};

void test_default_factory_registration() {
    const auto factory = runtime::activation::make_default_activation_codec_factory();
    expect(factory->supports("f32"), "default factory must register f32");
    expect(factory->supports("f16"), "default factory must register f16");
    expect(!factory->supports("missing"), "default factory accepted an unknown codec");

    try {
        (void) factory->create_codec("missing");
        fail("unknown activation encoding was accepted");
    } catch (const std::invalid_argument&) {
    }

    runtime::activation::ActivationCodecFactory invalid_factory;
    invalid_factory.register_codec("registered", [] {
        return std::make_shared<MisnamedCodec>();
    });
    try {
        (void) invalid_factory.create_codec("registered");
        fail("misnamed activation codec was accepted");
    } catch (const std::runtime_error&) {
    }
}

void test_f32_is_lossless_pass_through() {
    const auto codec =
        runtime::activation::make_default_activation_codec_factory()->create_codec("f32");
    const runtime::inference::Payload original = f32_activation();
    const runtime::inference::Payload encoded = codec->encode(original);
    const runtime::inference::Payload decoded = codec->decode(encoded);

    expect(encoded.tensor.dtype == "f32", "f32 codec changed the dtype");
    expect(encoded.bytes == original.bytes, "f32 codec changed activation bytes");
    expect(decoded.bytes == original.bytes, "f32 codec did not round-trip exactly");
}

void test_f16_halves_wire_bytes_and_decodes_to_f32() {
    const auto codec =
        runtime::activation::make_default_activation_codec_factory()->create_codec("f16");
    const runtime::inference::Payload original = f32_activation();
    const runtime::inference::Payload encoded = codec->encode(original);

    expect(encoded.tensor.dtype == "f16", "f16 codec did not set the wire dtype");
    expect(
        encoded.bytes.size() == original.bytes.size() / 2,
        "f16 codec did not halve activation bytes"
    );
    expect(
        runtime::inference::validate_payload(encoded).empty(),
        "f16 codec produced an invalid wire payload"
    );

    const runtime::inference::Payload decoded = codec->decode(encoded);
    expect(decoded.tensor.dtype == "f32", "f16 codec did not restore the engine dtype");
    expect(
        decoded.bytes.size() == original.bytes.size(),
        "f16 codec did not restore the f32 byte length"
    );
    expect(read_float(decoded.bytes, 0) == 0.0F, "positive zero changed");
    expect(std::signbit(read_float(decoded.bytes, 4)), "negative zero changed");
    expect(read_float(decoded.bytes, 8) == 1.0F, "one changed");
    expect(read_float(decoded.bytes, 12) == -2.0F, "negative two changed");
    expect(
        std::abs(read_float(decoded.bytes, 16) - 0.3333F) < 0.0003F,
        "rounded finite value exceeded f16 tolerance"
    );
    expect(read_float(decoded.bytes, 20) == 65504.0F, "maximum finite f16 changed");
}

void test_f16_handles_special_values_and_subnormals() {
    const auto codec =
        runtime::activation::make_default_activation_codec_factory()->create_codec("f16");
    runtime::inference::Payload original = f32_activation();
    original.bytes.clear();
    for (const float value : {
             std::numeric_limits<float>::infinity(),
             -std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN(),
             std::ldexp(1.0F, -14),
             std::ldexp(1.0F, -24),
             std::ldexp(1.0F, -25),
         }) {
        append_float(original.bytes, value);
    }

    const runtime::inference::Payload decoded = codec->decode(codec->encode(original));
    expect(std::isinf(read_float(decoded.bytes, 0)), "positive infinity changed");
    expect(read_float(decoded.bytes, 0) > 0, "positive infinity changed sign");
    expect(std::isinf(read_float(decoded.bytes, 4)), "negative infinity changed");
    expect(read_float(decoded.bytes, 4) < 0, "negative infinity changed sign");
    expect(std::isnan(read_float(decoded.bytes, 8)), "NaN changed to a finite value");
    expect(read_float(decoded.bytes, 12) == std::ldexp(1.0F, -14), "minimum normal changed");
    expect(
        read_float(decoded.bytes, 16) == std::ldexp(1.0F, -24),
        "minimum subnormal changed"
    );
    expect(read_float(decoded.bytes, 20) == 0.0F, "halfway underflow did not round to even");
}

} // namespace

int main() {
    test_default_factory_registration();
    test_f32_is_lossless_pass_through();
    test_f16_halves_wire_bytes_and_decodes_to_f32();
    test_f16_handles_special_values_and_subnormals();
    std::cout << "activation codec tests passed\n";
    return 0;
}
