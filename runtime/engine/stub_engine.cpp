#include "engine/stub_engine.hpp"

#include <ctime>
#include <sstream>
#include <utility>

namespace jetsonfabric::runtime {

StubEngine::StubEngine(std::string model, std::string mode)
    : model_(std::move(model)), mode_(std::move(mode)) {}

std::string StubEngine::runtime_name() const {
    return "jetsonfabric-runtime";
}

std::string StubEngine::mode() const {
    return mode_;
}

std::string StubEngine::model() const {
    return model_;
}

std::string StubEngine::chat_completion(const std::string& /*request_body*/) const {
    std::ostringstream body;
    body
        << "{"
        << "\"id\":\"chatcmpl-runtime-stub\","
        << "\"object\":\"chat.completion\","
        << "\"created\":" << std::time(nullptr) << ","
        << "\"model\":\"" << model_ << "\","
        << "\"choices\":["
        << "{"
        << "\"index\":0,"
        << "\"message\":{"
        << "\"role\":\"assistant\","
        << "\"content\":\"hello from jetsonfabric-runtime stub\""
        << "},"
        << "\"finish_reason\":\"stop\""
        << "}"
        << "],"
        << "\"usage\":{"
        << "\"prompt_tokens\":1,"
        << "\"completion_tokens\":6,"
        << "\"total_tokens\":7"
        << "}"
        << "}";
    return body.str();
}

} // namespace jetsonfabric::runtime
