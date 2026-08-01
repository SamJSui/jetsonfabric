#include "pipeline_parallel/continuous_batching_executor.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace jetsonfabric::runtime;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

inference::StageInput input_for(std::string session_id, inference::Phase phase) {
    inference::StageInput input;
    input.session_id = std::move(session_id);
    input.request_id = "request";
    input.model_id = "model";
    input.phase = phase;
    return input;
}

class RecordingExecutor final : public inference::Executor {
public:
    inference::ExecutionResult execute(const inference::StageInput&) const override {
        ++single_calls;
        return success(1);
    }

    std::vector<inference::ExecutionResult> execute_batch(
        const std::vector<inference::StageInput>& inputs
    ) const override {
        ++batch_calls;
        last_batch_size.store(static_cast<int>(inputs.size()));
        return std::vector<inference::ExecutionResult>(inputs.size(), success(inputs.size()));
    }

    static inference::ExecutionResult success(std::size_t batch_size) {
        inference::StageOutput output;
        output.execution_batch_size = static_cast<int>(batch_size);
        return inference::ExecutionResult::success(std::move(output));
    }

    mutable std::atomic_int single_calls{0};
    mutable std::atomic_int batch_calls{0};
    mutable std::atomic_int last_batch_size{0};
};

class FailingBatchExecutor final : public inference::Executor {
public:
    explicit FailingBatchExecutor(bool throw_error) : throw_error_(throw_error) {}

    inference::ExecutionResult execute(const inference::StageInput&) const override {
        return RecordingExecutor::success(1);
    }

    std::vector<inference::ExecutionResult> execute_batch(
        const std::vector<inference::StageInput>&
    ) const override {
        if (throw_error_) {
            throw std::runtime_error("injected batch failure");
        }
        return {};
    }

private:
    bool throw_error_;
};

void expect_batch_failure(bool throw_error, const std::string& expected_code) {
    pipeline_parallel::ContinuousBatchingExecutor executor(
        std::make_unique<FailingBatchExecutor>(throw_error),
        pipeline_parallel::ContinuousBatchingConfig{
            .max_batch_size = 2,
            .max_wait = std::chrono::microseconds::zero(),
        }
    );
    const inference::ExecutionResult result = executor.execute(
        input_for("failure", inference::Phase::Decode)
    );
    expect(!result.ok && result.error.code == expected_code, "unexpected batch failure result");
}

} // namespace

int main() {
    using namespace jetsonfabric::runtime;

    auto delegate = std::make_unique<RecordingExecutor>();
    RecordingExecutor* observer = delegate.get();
    pipeline_parallel::ContinuousBatchingExecutor executor(
        std::move(delegate),
        pipeline_parallel::ContinuousBatchingConfig{
            .max_batch_size = 2,
            .max_wait = std::chrono::milliseconds(20),
        }
    );

    const inference::ExecutionResult prefill = executor.execute(
        input_for("prefill", inference::Phase::Prefill)
    );
    expect(prefill.ok && observer->single_calls == 1, "prefill did not bypass decode batching");

    auto first = std::async(std::launch::async, [&executor]() {
        return executor.execute(input_for("session-a", inference::Phase::Decode));
    });
    auto second = std::async(std::launch::async, [&executor]() {
        return executor.execute(input_for("session-b", inference::Phase::Decode));
    });

    const inference::ExecutionResult first_result = first.get();
    const inference::ExecutionResult second_result = second.get();
    expect(first_result.ok && second_result.ok, "batched decode failed");
    expect(observer->batch_calls == 1, "concurrent decode steps were not coalesced");
    expect(observer->last_batch_size == 2, "decode executor received the wrong batch size");
    expect(
        first_result.output.execution_batch_size == 2 &&
            second_result.output.execution_batch_size == 2,
        "batch-size telemetry did not reach every result"
    );

    expect_batch_failure(true, "continuous_batch_failed");
    expect_batch_failure(false, "continuous_batch_result_mismatch");

    std::cout << "continuous batching executor tests passed\n";
    return 0;
}
