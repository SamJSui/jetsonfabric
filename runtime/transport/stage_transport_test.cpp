#include "transport/stage_transport.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

class RecordingTransport final : public jetsonfabric::runtime::transport::StageTransport {
public:
    jetsonfabric::runtime::pipeline_parallel::StageRunResult invoke(
        const jetsonfabric::runtime::protocol::GenerationStage& stage,
        const jetsonfabric::runtime::protocol::StageRequest& request,
        jetsonfabric::runtime::pipeline_parallel::StageOperation operation
    ) const override {
        invoked = true;
        stage_index = stage.stage_index;
        request_id = request.request_id;
        close_session = operation ==
            jetsonfabric::runtime::pipeline_parallel::StageOperation::CloseSession;

        jetsonfabric::runtime::pipeline_parallel::StageRunResult result;
        result.ok = true;
        return result;
    }

    mutable bool invoked = false;
    mutable int stage_index = -1;
    mutable std::string request_id;
    mutable bool close_session = false;
};

} // namespace

int main() {
    using namespace jetsonfabric::runtime;

    RecordingTransport transport;
    protocol::GenerationStage stage;
    stage.stage_index = 1;
    protocol::StageRequest request;
    request.request_id = "request-1";

    const pipeline_parallel::StageRunResult result = transport.invoke(
        stage,
        request,
        pipeline_parallel::StageOperation::CloseSession
    );

    expect(result.ok, "transport result was not returned");
    expect(transport.invoked, "transport implementation was not invoked");
    expect(transport.stage_index == 1, "stage metadata did not cross transport boundary");
    expect(transport.request_id == "request-1", "request metadata did not cross transport boundary");
    expect(transport.close_session, "stage operation did not cross transport boundary");

    std::cout << "stage transport tests passed\n";
    return 0;
}
