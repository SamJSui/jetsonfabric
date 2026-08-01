#include "activation/activation_codec_factory.hpp"

#include <bit>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::activation {
namespace {

constexpr const char* kF32 = "f32";
constexpr const char* kF16 = "f16";

void require_activation(const inference::Payload& payload) {
    if (payload.kind != inference::PayloadKind::Activation) {
        throw std::invalid_argument("activation codec requires an activation payload");
    }
}

void require_dtype(const inference::Payload& payload, const char* expected) {
    if (payload.tensor.dtype != expected) {
        throw std::invalid_argument(
            "activation codec expected " + std::string(expected) +
            " input, got " + payload.tensor.dtype
        );
    }
}

std::uint32_t read_u32_le(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
}

void append_u16_le(std::vector<std::uint8_t>& output, std::uint16_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xffU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
}

std::uint16_t float_to_half(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    std::uint32_t mantissa = bits & 0x7fffffU;

    if (exponent == 0xffU) {
        if (mantissa == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00U);
        }
        const std::uint16_t payload = static_cast<std::uint16_t>(mantissa >> 13U);
        return static_cast<std::uint16_t>(sign | 0x7c00U | payload | 0x0200U);
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000U;
        const unsigned shift = static_cast<unsigned>(14 - half_exponent);
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t remainder_mask = (1U << shift) - 1U;
        const std::uint32_t remainder = mantissa & remainder_mask;
        const std::uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1U) != 0)) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U ||
        (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00U);
            }
        }
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint16_t>(half_exponent) << 10U) |
        static_cast<std::uint16_t>(half_mantissa)
    );
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1fU;
    std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int normalized_exponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --normalized_exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign |
                (static_cast<std::uint32_t>(normalized_exponent + 127) << 23U) |
                (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign |
            ((exponent + static_cast<std::uint32_t>(127 - 15)) << 23U) |
            (mantissa << 13U);
    }

    return std::bit_cast<float>(bits);
}

class F32ActivationCodec final : public ActivationCodec {
public:
    const std::string& name() const noexcept override {
        static const std::string value = kF32;
        return value;
    }

    inference::Payload encode(inference::Payload payload) const override {
        require_activation(payload);
        require_dtype(payload, kF32);
        return payload;
    }

    inference::Payload decode(inference::Payload payload) const override {
        require_activation(payload);
        require_dtype(payload, kF32);
        return payload;
    }
};

class F16ActivationCodec final : public ActivationCodec {
public:
    const std::string& name() const noexcept override {
        static const std::string value = kF16;
        return value;
    }

    inference::Payload encode(inference::Payload payload) const override {
        require_activation(payload);
        require_dtype(payload, kF32);
        if (payload.bytes.size() % sizeof(float) != 0) {
            throw std::invalid_argument("f32 activation byte length must be divisible by four");
        }

        std::vector<std::uint8_t> encoded;
        encoded.reserve(payload.bytes.size() / 2);
        for (std::size_t offset = 0; offset < payload.bytes.size(); offset += sizeof(float)) {
            const float value = std::bit_cast<float>(read_u32_le(payload.bytes.data() + offset));
            append_u16_le(encoded, float_to_half(value));
        }
        payload.tensor.dtype = kF16;
        payload.bytes = std::move(encoded);
        return payload;
    }

    inference::Payload decode(inference::Payload payload) const override {
        require_activation(payload);
        require_dtype(payload, kF16);
        if (payload.bytes.size() % sizeof(std::uint16_t) != 0) {
            throw std::invalid_argument("f16 activation byte length must be divisible by two");
        }

        std::vector<std::uint8_t> decoded;
        decoded.reserve(payload.bytes.size() * 2);
        for (std::size_t offset = 0; offset < payload.bytes.size(); offset += 2) {
            const std::uint16_t value = static_cast<std::uint16_t>(payload.bytes[offset]) |
                static_cast<std::uint16_t>(payload.bytes[offset + 1] << 8U);
            append_u32_le(decoded, std::bit_cast<std::uint32_t>(half_to_float(value)));
        }
        payload.tensor.dtype = kF32;
        payload.bytes = std::move(decoded);
        return payload;
    }
};

} // namespace

std::shared_ptr<const ActivationCodecFactory> make_default_activation_codec_factory() {
    auto factory = std::make_shared<ActivationCodecFactory>();
    factory->register_codec(kF32, [] {
        return std::make_shared<F32ActivationCodec>();
    });
    factory->register_codec(kF16, [] {
        return std::make_shared<F16ActivationCodec>();
    });
    return factory;
}

} // namespace jetsonfabric::runtime::activation
