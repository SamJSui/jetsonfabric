#include "protocol/stage_control.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace jetsonfabric::runtime::protocol {
namespace {

constexpr std::size_t kHeaderSize = 20;

std::uint32_t read_u32_be(const char* data) {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(data[0])) << 24U) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(data[1])) << 16U) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(data[2])) << 8U) |
           static_cast<std::uint32_t>(static_cast<unsigned char>(data[3]));
}

} // namespace

std::string decode_stage_operation(const std::string& frame) {
    if (frame.size() < kHeaderSize) {
        throw std::invalid_argument("truncated stagewire header");
    }
    const std::uint32_t metadata_length = read_u32_be(frame.data() + 8);
    if (metadata_length == 0 || kHeaderSize + static_cast<std::size_t>(metadata_length) > frame.size()) {
        throw std::invalid_argument("invalid stagewire metadata length");
    }
    const std::string metadata_text = frame.substr(kHeaderSize, metadata_length);
    const nlohmann::json metadata = nlohmann::json::parse(metadata_text);
    const auto found = metadata.find("operation");
    const std::string operation = found == metadata.end() || found->is_null()
        ? kStageOperationExecute
        : found->get<std::string>();
    if (operation != kStageOperationExecute && operation != kStageOperationCloseSession) {
        throw std::invalid_argument("invalid stage operation: " + operation);
    }
    return operation;
}

} // namespace jetsonfabric::runtime::protocol
