#include "pipeline_parallel/continuous_batching_executor.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::pipeline_parallel {

struct ContinuousBatchingExecutor::Impl {
    struct PendingExecution {
        explicit PendingExecution(inference::StageInput input_in)
            : input(std::move(input_in)) {}

        inference::StageInput input;
        std::promise<inference::ExecutionResult> completion;
    };

    Impl(std::unique_ptr<inference::Executor> delegate_in, ContinuousBatchingConfig config_in)
        : delegate(std::move(delegate_in)),
          config(config_in) {
        if (!delegate) {
            throw std::invalid_argument("continuous batching requires a layer executor");
        }
        if (config.max_batch_size < 2U) {
            throw std::invalid_argument("continuous batching size must be at least two");
        }
        if (config.max_wait < std::chrono::microseconds::zero()) {
            throw std::invalid_argument("continuous batching wait must not be negative");
        }
        worker = std::jthread([this](std::stop_token stop_token) { run(stop_token); });
    }

    ~Impl() {
        worker.request_stop();
        ready.notify_all();
    }

    inference::ExecutionResult execute(const inference::StageInput& input) const {
        if (input.phase == inference::Phase::Prefill) {
            const std::lock_guard delegate_lock(delegate_mutex);
            return delegate->execute(input);
        }

        auto pending = std::make_shared<PendingExecution>(input);
        std::future<inference::ExecutionResult> result = pending->completion.get_future();
        {
            const std::lock_guard lock(mutex);
            queue.push_back(std::move(pending));
        }
        ready.notify_one();
        return result.get();
    }

    void run(std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
            std::vector<std::shared_ptr<PendingExecution>> pending = next_batch(stop_token);
            if (pending.empty()) {
                continue;
            }

            std::vector<inference::StageInput> inputs;
            inputs.reserve(pending.size());
            for (const auto& execution : pending) {
                inputs.push_back(execution->input);
            }

            std::vector<inference::ExecutionResult> results;
            try {
                const std::lock_guard delegate_lock(delegate_mutex);
                results = delegate->execute_batch(inputs);
            } catch (const std::exception& error) {
                results.assign(
                    inputs.size(),
                    inference::ExecutionResult::failure(
                        "continuous_batch_failed",
                        error.what()
                    )
                );
            }
            if (results.size() != pending.size()) {
                results.assign(
                    pending.size(),
                    inference::ExecutionResult::failure(
                        "continuous_batch_result_mismatch",
                        "layer executor returned the wrong number of batch results"
                    )
                );
            }
            for (std::size_t index = 0; index < pending.size(); ++index) {
                pending[index]->completion.set_value(std::move(results[index]));
            }
        }
        fail_pending();
    }

    std::vector<std::shared_ptr<PendingExecution>> next_batch(std::stop_token stop_token) {
        std::unique_lock lock(mutex);
        ready.wait(lock, [this, &stop_token]() {
            return stop_token.stop_requested() || !queue.empty();
        });
        if (stop_token.stop_requested()) {
            return {};
        }

        const auto deadline = std::chrono::steady_clock::now() + config.max_wait;
        ready.wait_until(lock, deadline, [this, &stop_token]() {
            return stop_token.stop_requested() || queue.size() >= config.max_batch_size;
        });
        if (stop_token.stop_requested()) {
            return {};
        }

        const std::size_t count = std::min(config.max_batch_size, queue.size());
        std::vector<std::shared_ptr<PendingExecution>> batch;
        batch.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            batch.push_back(std::move(queue.front()));
            queue.pop_front();
        }
        return batch;
    }

    void fail_pending() {
        std::deque<std::shared_ptr<PendingExecution>> abandoned;
        {
            const std::lock_guard lock(mutex);
            abandoned.swap(queue);
        }
        for (const auto& pending : abandoned) {
            pending->completion.set_value(inference::ExecutionResult::failure(
                "continuous_batch_stopped",
                "continuous batching executor stopped before execution"
            ));
        }
    }

    std::unique_ptr<inference::Executor> delegate;
    ContinuousBatchingConfig config;
    mutable std::mutex mutex;
    mutable std::mutex delegate_mutex;
    mutable std::condition_variable ready;
    mutable std::deque<std::shared_ptr<PendingExecution>> queue;
    std::jthread worker;
};

ContinuousBatchingExecutor::ContinuousBatchingExecutor(
    std::unique_ptr<inference::Executor> delegate,
    ContinuousBatchingConfig config
)
    : impl_(std::make_unique<Impl>(std::move(delegate), config)) {}

ContinuousBatchingExecutor::~ContinuousBatchingExecutor() = default;

inference::ExecutionResult ContinuousBatchingExecutor::execute(
    const inference::StageInput& input
) const {
    return impl_->execute(input);
}

std::vector<inference::ExecutionResult> ContinuousBatchingExecutor::execute_batch(
    const std::vector<inference::StageInput>& inputs
) const {
    const std::lock_guard lock(impl_->delegate_mutex);
    return impl_->delegate->execute_batch(inputs);
}

void ContinuousBatchingExecutor::close_session(const std::string& session_id) const {
    const std::lock_guard lock(impl_->delegate_mutex);
    impl_->delegate->close_session(session_id);
}

void ContinuousBatchingExecutor::rollback_session(
    const std::string& session_id,
    int token_count
) const {
    const std::lock_guard lock(impl_->delegate_mutex);
    impl_->delegate->rollback_session(session_id, token_count);
}

} // namespace jetsonfabric::runtime::pipeline_parallel
