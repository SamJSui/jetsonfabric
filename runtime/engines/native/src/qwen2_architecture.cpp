#include "jetsonfabric/native_inference.hpp"

#include "model_architecture.hpp"
#include "qwen2_metadata.hpp"
#include "tensor_store.hpp"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace jetsonfabric::native {
namespace {

constexpr std::size_t kGraphSize = 8192;

struct LayerCache {
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
};

enum class OutputKind { Logits, GreedyToken };
enum class InputKind { Prefill, Decode };

struct GraphTimings {
    double allocation_ms = 0.0;
    double host_input_preparation_ms = 0.0;
    double compute_ms = 0.0;
    double output_read_ms = 0.0;
};

void require_shape(
    const TensorStore& tensors,
    const std::string& name,
    std::initializer_list<std::int64_t> expected
) {
    const ggml_tensor * tensor = tensors.require(name);
    if (ggml_n_dims(tensor) != static_cast<int>(expected.size())) {
        throw std::runtime_error("native Qwen2 tensor has wrong rank: " + name);
    }
    std::size_t dimension = 0;
    for (const std::int64_t size : expected) {
        if (tensor->ne[dimension++] != size) {
            throw std::runtime_error("native Qwen2 tensor has wrong shape: " + name);
        }
    }
}

void validate_qwen2_tensors(const TensorStore& tensors, const Qwen2HParams& params) {
    const std::int64_t embedding = params.public_info.embedding_length;
    const std::int64_t head_length = embedding / params.head_count;
    const std::int64_t kv_length = head_length * params.kv_head_count;
    const std::int64_t feed_forward = params.feed_forward_length;
    const std::int64_t vocabulary = tensors.vocabulary_size();
    require_shape(tensors, "token_embd.weight", {embedding, vocabulary});
    require_shape(tensors, "output_norm.weight", {embedding});
    require_shape(tensors, "output.weight", {embedding, vocabulary});
    if (tensors.find("output.bias") != nullptr) {
        require_shape(tensors, "output.bias", {vocabulary});
    }
    for (std::uint32_t layer = 0; layer < params.public_info.layer_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        require_shape(tensors, prefix + "attn_norm.weight", {embedding});
        require_shape(tensors, prefix + "attn_q.weight", {embedding, embedding});
        require_shape(tensors, prefix + "attn_q.bias", {embedding});
        require_shape(tensors, prefix + "attn_k.weight", {embedding, kv_length});
        require_shape(tensors, prefix + "attn_k.bias", {kv_length});
        require_shape(tensors, prefix + "attn_v.weight", {embedding, kv_length});
        require_shape(tensors, prefix + "attn_v.bias", {kv_length});
        require_shape(tensors, prefix + "attn_output.weight", {embedding, embedding});
        require_shape(tensors, prefix + "ffn_norm.weight", {embedding});
        require_shape(tensors, prefix + "ffn_gate.weight", {embedding, feed_forward});
        require_shape(tensors, prefix + "ffn_up.weight", {embedding, feed_forward});
        require_shape(tensors, prefix + "ffn_down.weight", {feed_forward, embedding});
    }
}

class ForwardGraph {
public:
    ForwardGraph(
        TensorStore& weights,
        const Qwen2HParams& params,
        std::span<const LayerCache> cache,
        ggml_backend_sched_t scheduler,
        std::size_t attention_length,
        std::size_t token_count,
        OutputKind output_kind,
        InputKind input_kind
    ) : weights_(weights), params_(params), cache_(cache), scheduler_(scheduler),
        attention_length_(attention_length), token_count_(token_count),
        output_kind_(output_kind), input_kind_(input_kind) {
        if (token_count == 0 || attention_length < token_count ||
            attention_length > params.public_info.context_length ||
            cache.size() != params.public_info.layer_count || scheduler == nullptr) {
            throw std::invalid_argument("native Qwen2 token count is outside model context");
        }
        const std::size_t metadata_bytes =
            20000U * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphSize, false);
        context_.reset(ggml_init(ggml_init_params{
            .mem_size = metadata_bytes,
            .mem_buffer = nullptr,
            .no_alloc = true,
        }));
        if (!context_) {
            throw std::runtime_error("could not allocate native Qwen2 graph metadata");
        }
        build();
        assign_backends();
        if (input_kind_ == InputKind::Prefill) {
            prepare_fixed_inputs();
        }
    }

    std::vector<float> compute_logits(
        std::span<const std::int32_t> tokens,
        std::size_t position
    ) {
        if (output_kind_ != OutputKind::Logits) {
            throw std::logic_error("native Qwen2 graph does not produce logits");
        }
        compute(tokens, position);
        std::vector<float> output(static_cast<std::size_t>(logits_->ne[0]));
        ggml_backend_tensor_get(logits_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

    std::int32_t compute_greedy_token(
        std::span<const std::int32_t> tokens,
        std::size_t position,
        GraphTimings * timings = nullptr
    ) {
        if (output_kind_ != OutputKind::GreedyToken) {
            throw std::logic_error("native Qwen2 graph does not produce a greedy token");
        }
        compute(tokens, position, timings);
        const auto read_start = std::chrono::steady_clock::now();
        std::int32_t token = -1;
        ggml_backend_tensor_get(greedy_token_, &token, 0, sizeof(token));
        if (timings != nullptr) {
            timings->output_read_ms = elapsed_ms(read_start);
        }
        return token;
    }

    std::uint64_t host_input_bytes() const {
        return fixed_positions_.size() * sizeof(std::int32_t) +
            fixed_attention_mask_.size() * sizeof(float) +
            fixed_value_cache_indices_.size() * sizeof(std::int32_t) +
            sizeof(fixed_output_index_);
    }

private:
    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count();
    }

    void compute(
        std::span<const std::int32_t> tokens,
        std::size_t position,
        GraphTimings * timings = nullptr
    ) {
        if (!allocated_) {
            const auto allocation_start = std::chrono::steady_clock::now();
            ggml_backend_sched_reset(scheduler_);
            if (!ggml_backend_sched_alloc_graph(scheduler_, graph_)) {
                throw std::runtime_error("could not allocate native Qwen2 compute graph");
            }
            allocated_ = true;
            if (timings != nullptr) {
                timings->allocation_ms = elapsed_ms(allocation_start);
            }
        }
        const auto input_start = std::chrono::steady_clock::now();
        set_inputs(tokens, position);
        if (timings != nullptr) {
            timings->host_input_preparation_ms = elapsed_ms(input_start);
        }
        const auto compute_start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_sched_graph_compute(scheduler_, graph_);
        if (timings != nullptr) {
            timings->compute_ms = elapsed_ms(compute_start);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("native Qwen2 graph computation failed");
        }
    }

    void assign_backends() {
        const ggml_backend_t input_backend = weights_.input_backend();
        ggml_backend_sched_set_tensor_backend(scheduler_, input_tokens_, input_backend);
        ggml_backend_sched_set_tensor_backend(scheduler_, positions_, input_backend);
        ggml_backend_sched_set_tensor_backend(scheduler_, mask_, input_backend);
        ggml_backend_sched_set_tensor_backend(scheduler_, output_index_, input_backend);
        ggml_backend_sched_set_tensor_backend(scheduler_, cache_indices_, input_backend);
        ggml_backend_sched_set_tensor_backend(
            scheduler_, value_cache_indices_, input_backend
        );

        const ggml_backend_t compute_backend = weights_.backend();
        ggml_backend_sched_set_tensor_backend(scheduler_, logits_, compute_backend);
        if (greedy_token_ != nullptr) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_, greedy_token_, compute_backend
            );
        }
    }

    ggml_tensor * multiply(ggml_tensor * left, ggml_tensor * right, const char * label) {
        const bool compatible = left->ne[0] == right->ne[0] &&
            right->ne[2] % left->ne[2] == 0 && right->ne[3] % left->ne[3] == 0;
        if (!compatible) {
            std::ostringstream message;
            message << label << " cannot multiply ["
                    << left->ne[0] << ',' << left->ne[1] << ',' << left->ne[2]
                    << "] by [" << right->ne[0] << ',' << right->ne[1] << ','
                    << right->ne[2] << ']';
            throw std::runtime_error(message.str());
        }
        return ggml_mul_mat(context_.get(), left, right);
    }

    ggml_tensor * linear(
        const std::string& weight_name,
        ggml_tensor * input,
        const std::string& bias_name = {}
    ) {
        ggml_tensor * output = multiply(weights_.require(weight_name), input, weight_name.c_str());
        if (!bias_name.empty()) {
            output = ggml_add(context_.get(), output, weights_.require(bias_name));
        }
        return output;
    }

    ggml_tensor * rms_norm(ggml_tensor * input, const std::string& weight_name) {
        ggml_tensor * normalized = ggml_rms_norm(context_.get(), input, params_.rms_epsilon);
        return ggml_mul(context_.get(), normalized, weights_.require(weight_name));
    }

    ggml_tensor * output_projection(ggml_tensor * input) {
        ggml_tensor * output = linear("output.weight", input);
        if (ggml_tensor * bias = weights_.find("output.bias")) {
            output = ggml_add(context_.get(), output, bias);
        }
        return output;
    }

    ggml_tensor * attention(ggml_tensor * input, std::uint32_t layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const ModelInfo& info = params_.public_info;
        const std::int64_t tokens = static_cast<std::int64_t>(token_count_);
        const std::int64_t key_count = static_cast<std::int64_t>(attention_length_);
        const std::int64_t head_length = info.embedding_length / params_.head_count;
        const std::int64_t kv_length = head_length * params_.kv_head_count;

        ggml_tensor * query = linear(prefix + "attn_q.weight", input, prefix + "attn_q.bias");
        ggml_tensor * key = linear(prefix + "attn_k.weight", input, prefix + "attn_k.bias");
        ggml_tensor * value = linear(prefix + "attn_v.weight", input, prefix + "attn_v.bias");
        query = ggml_reshape_3d(context_.get(), query, head_length, params_.head_count, tokens);
        key = ggml_reshape_3d(context_.get(), key, head_length, params_.kv_head_count, tokens);
        value = ggml_reshape_3d(context_.get(), value, head_length, params_.kv_head_count, tokens);

        query = apply_rope(query);
        key = apply_rope(key);
        query = ggml_permute(context_.get(), query, 0, 2, 1, 3);

        const LayerCache& layer_cache = cache_[layer];
        key = ggml_view_2d(
            context_.get(), key, kv_length, tokens, key->nb[2], 0
        );
        value = ggml_reshape_2d(
            context_.get(), value, 1, ggml_nelements(value)
        );
        ggml_tensor * key_write = ggml_set_rows(
            context_.get(), layer_cache.key, key, cache_indices_
        );
        ggml_tensor * value_destination = ggml_reshape_2d(
            context_.get(), layer_cache.value, 1, ggml_nelements(layer_cache.value)
        );
        ggml_tensor * value_write = ggml_set_rows(
            context_.get(), value_destination, value, value_cache_indices_
        );
        ggml_tensor * updated_value_cache = ggml_reshape_3d(
            context_.get(), value_write,
            layer_cache.value->ne[0], layer_cache.value->ne[1], layer_cache.value->ne[2]
        );

        key = ggml_view_3d(
            context_.get(), key_write,
            head_length, params_.kv_head_count, key_count,
            ggml_row_size(layer_cache.key->type, head_length),
            ggml_row_size(layer_cache.key->type, kv_length), 0
        );
        value = ggml_view_3d(
            context_.get(), updated_value_cache,
            key_count, head_length, params_.kv_head_count,
            layer_cache.value->nb[1], layer_cache.value->nb[2], 0
        );
        key = ggml_permute(context_.get(), key, 0, 2, 1, 3);
        ggml_tensor * scores = multiply(key, query, "attention scores");
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(
            context_.get(), scores, mask_,
            1.0F / std::sqrt(static_cast<float>(head_length)), 0.0F
        );
        ggml_tensor * attended = multiply(value, scores, "attention values");
        attended = ggml_permute(context_.get(), attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(
            context_.get(), attended, info.embedding_length, tokens
        );
        if (attended->ne[0] != info.embedding_length || value->ne[0] != key_count ||
            value->ne[1] != head_length || key->ne[0] != head_length ||
            key->ne[1] != key_count || key->ne[2] != params_.kv_head_count || kv_length <= 0) {
            throw std::runtime_error("native Qwen2 attention graph has inconsistent dimensions");
        }
        return linear(prefix + "attn_output.weight", attended);
    }

    ggml_tensor * apply_rope(ggml_tensor * tensor) {
        return ggml_rope_ext(
            context_.get(), tensor, positions_, nullptr,
            params_.rope_dimension_count, GGML_ROPE_TYPE_NEOX,
            params_.public_info.context_length,
            params_.rope_frequency_base, 1.0F, 0.0F, 1.0F, 32.0F, 1.0F
        );
    }

    ggml_tensor * feed_forward(ggml_tensor * input, std::uint32_t layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        ggml_tensor * up = linear(prefix + "ffn_up.weight", input);
        ggml_tensor * gate = linear(prefix + "ffn_gate.weight", input);
        gate = ggml_silu(context_.get(), gate);
        return linear(prefix + "ffn_down.weight", ggml_mul(context_.get(), gate, up));
    }

    void build() {
        const ModelInfo& info = params_.public_info;
        const std::int64_t key_count = static_cast<std::int64_t>(attention_length_);
        input_tokens_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32, static_cast<std::int64_t>(token_count_)
        );
        positions_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32, static_cast<std::int64_t>(token_count_)
        );
        mask_ = ggml_new_tensor_2d(
            context_.get(), GGML_TYPE_F32,
            key_count,
            static_cast<std::int64_t>(token_count_)
        );
        output_index_ = ggml_new_tensor_1d(context_.get(), GGML_TYPE_I32, 1);
        cache_indices_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32, static_cast<std::int64_t>(token_count_)
        );
        value_cache_indices_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32,
            static_cast<std::int64_t>(token_count_) * info.embedding_length /
                params_.head_count * params_.kv_head_count
        );
        ggml_set_input(input_tokens_);
        ggml_set_input(positions_);
        ggml_set_input(mask_);
        ggml_set_input(output_index_);
        ggml_set_input(cache_indices_);
        ggml_set_input(value_cache_indices_);
        graph_ = ggml_new_graph_custom(context_.get(), kGraphSize, false);

        ggml_tensor * hidden = ggml_get_rows(
            context_.get(), weights_.require("token_embd.weight"), input_tokens_
        );
        for (std::uint32_t layer = 0; layer < info.layer_count; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            ggml_tensor * attention_input = hidden;
            ggml_tensor * normalized = rms_norm(hidden, prefix + "attn_norm.weight");
            hidden = ggml_add(context_.get(), attention(normalized, layer), attention_input);
            ggml_tensor * ffn_input = hidden;
            normalized = rms_norm(hidden, prefix + "ffn_norm.weight");
            hidden = ggml_add(context_.get(), feed_forward(normalized, layer), ffn_input);
        }
        hidden = ggml_get_rows(context_.get(), hidden, output_index_);
        hidden = rms_norm(hidden, "output_norm.weight");
        logits_ = output_projection(hidden);
        if (logits_->ne[0] <= 0) {
            throw std::runtime_error("native Qwen2 logits have invalid vocabulary dimension");
        }
        if (output_kind_ == OutputKind::Logits) {
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
        } else {
            greedy_token_ = ggml_argmax(context_.get(), logits_);
            ggml_set_output(greedy_token_);
            ggml_build_forward_expand(graph_, greedy_token_);
        }
    }

    void set_inputs(std::span<const std::int32_t> tokens, std::size_t position) {
        if (tokens.size() != token_count_ || position + token_count_ > attention_length_) {
            throw std::invalid_argument("native Qwen2 graph token count changed");
        }
        if (input_kind_ == InputKind::Prefill && position != 0) {
            throw std::invalid_argument("native Qwen2 prefill graph must start at position zero");
        }
        ggml_backend_tensor_set(input_tokens_, tokens.data(), 0, tokens.size_bytes());
        if (input_kind_ == InputKind::Prefill) {
            upload_fixed_inputs();
            return;
        }
        std::vector<std::int32_t> positions(token_count_);
        for (std::size_t query = 0; query < token_count_; ++query) {
            positions[query] = static_cast<std::int32_t>(position + query);
        }
        const std::vector<float> mask = build_attention_mask(position);
        const std::vector<std::int32_t> value_cache_indices =
            build_value_cache_indices(position);
        const std::int32_t output_index = static_cast<std::int32_t>(token_count_ - 1);
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_tensor_set(output_index_, &output_index, 0, sizeof(output_index));
        ggml_backend_tensor_set(
            cache_indices_, positions.data(), 0, positions.size() * sizeof(std::int32_t)
        );
        ggml_backend_tensor_set(
            value_cache_indices_, value_cache_indices.data(), 0,
            value_cache_indices.size() * sizeof(std::int32_t)
        );
    }

    void prepare_fixed_inputs() {
        fixed_positions_.resize(token_count_);
        for (std::size_t query = 0; query < token_count_; ++query) {
            fixed_positions_[query] = static_cast<std::int32_t>(query);
        }
        fixed_attention_mask_ = build_attention_mask(0);
        fixed_value_cache_indices_ = build_value_cache_indices(0);
        fixed_output_index_ = static_cast<std::int32_t>(token_count_ - 1);
    }

    void upload_fixed_inputs() {
        ggml_backend_tensor_set(
            positions_, fixed_positions_.data(), 0,
            fixed_positions_.size() * sizeof(std::int32_t)
        );
        ggml_backend_tensor_set(
            mask_, fixed_attention_mask_.data(), 0,
            fixed_attention_mask_.size() * sizeof(float)
        );
        ggml_backend_tensor_set(
            output_index_, &fixed_output_index_, 0, sizeof(fixed_output_index_)
        );
        ggml_backend_tensor_set(
            cache_indices_, fixed_positions_.data(), 0,
            fixed_positions_.size() * sizeof(std::int32_t)
        );
        ggml_backend_tensor_set(
            value_cache_indices_, fixed_value_cache_indices_.data(), 0,
            fixed_value_cache_indices_.size() * sizeof(std::int32_t)
        );
    }

    std::vector<float> build_attention_mask(std::size_t position) const {
        std::vector<float> mask(token_count_ * attention_length_);
        for (std::size_t query = 0; query < token_count_; ++query) {
            const std::size_t absolute_query = position + query;
            for (std::size_t key = 0; key < attention_length_; ++key) {
                mask[query * attention_length_ + key] = key > absolute_query
                    ? -std::numeric_limits<float>::infinity()
                    : 0.0F;
            }
        }
        return mask;
    }

    std::vector<std::int32_t> build_value_cache_indices(std::size_t position) const {
        const std::size_t head_length =
            params_.public_info.embedding_length / params_.head_count;
        const std::size_t kv_length = head_length * params_.kv_head_count;
        std::vector<std::int32_t> indices(token_count_ * kv_length);
        for (std::size_t token = 0; token < token_count_; ++token) {
            for (std::size_t head = 0; head < params_.kv_head_count; ++head) {
                for (std::size_t element = 0; element < head_length; ++element) {
                    const std::size_t source =
                        token * kv_length + head * head_length + element;
                    const std::size_t destination = position + token +
                        capacity_stride() * (head * head_length + element);
                    indices[source] = static_cast<std::int32_t>(destination);
                }
            }
        }
        return indices;
    }

    std::size_t capacity_stride() const { return cache_.front().value->ne[0]; }

    TensorStore& weights_;
    const Qwen2HParams& params_;
    std::span<const LayerCache> cache_;
    ggml_backend_sched_t scheduler_;
    std::size_t attention_length_;
    std::size_t token_count_;
    OutputKind output_kind_;
    InputKind input_kind_;
    bool allocated_ = false;
    ggml_context_ptr context_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_tokens_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_index_ = nullptr;
    ggml_tensor * cache_indices_ = nullptr;
    ggml_tensor * value_cache_indices_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_tensor * greedy_token_ = nullptr;
    std::vector<std::int32_t> fixed_positions_;
    std::vector<float> fixed_attention_mask_;
    std::vector<std::int32_t> fixed_value_cache_indices_;
    std::int32_t fixed_output_index_ = 0;
};

} // namespace

class Qwen2Session final : public InferenceSession {
public:
    Qwen2Session(
        TensorStore& tensors,
        Qwen2HParams hparams,
        std::size_t capacity
    ) : tensors_(tensors), hparams_(std::move(hparams)), capacity_(capacity) {
        if (capacity == 0 || capacity > hparams_.public_info.context_length) {
            throw std::invalid_argument("native Qwen2 cache capacity is outside model context");
        }
        allocate_cache();
        prefill_scheduler_ = create_scheduler();
        decode_scheduler_ = create_scheduler();
    }

    std::size_t capacity() const override { return capacity_; }
    void reset() override {
        position_ = 0;
        decode_greedy_graph_.reset();
    }

    std::vector<float> prefill_logits(
        std::span<const std::int32_t> tokens
    ) override {
        validate_prefill(tokens);
        ggml_backend_sched_ptr logits_scheduler = create_scheduler();
        std::vector<float> output = ForwardGraph(
            tensors_, hparams_, cache_, logits_scheduler.get(),
            tokens.size(), tokens.size(), OutputKind::Logits,
            InputKind::Prefill
        ).compute_logits(tokens, 0);
        position_ += tokens.size();
        return output;
    }

    PrefillResult prefill_greedy(
        std::span<const std::int32_t> tokens
    ) override {
        validate_prefill(tokens);
        const bool graph_reused = prefill_greedy_graph_ != nullptr &&
            prefill_token_count_ == tokens.size();
        double graph_build_ms = 0.0;
        if (!graph_reused) {
            const auto build_start = std::chrono::steady_clock::now();
            prefill_greedy_graph_ = std::make_unique<ForwardGraph>(
                tensors_, hparams_, cache_, prefill_scheduler_.get(),
                tokens.size(), tokens.size(), OutputKind::GreedyToken,
                InputKind::Prefill
            );
            graph_build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - build_start
            ).count();
            prefill_token_count_ = tokens.size();
        }
        GraphTimings graph_timings;
        const std::int32_t token = prefill_greedy_graph_->compute_greedy_token(
            tokens, 0, &graph_timings
        );
        position_ += tokens.size();
        return PrefillResult{
            .token = token,
            .metrics = PrefillMetrics{
                .plan_reused = graph_reused,
                .planning_ms = graph_build_ms,
                .allocation_ms = graph_timings.allocation_ms,
                .host_input_preparation_ms = graph_timings.host_input_preparation_ms,
                .compute_ms = graph_timings.compute_ms,
                .output_read_ms = graph_timings.output_read_ms,
            },
        };
    }

    std::int32_t decode_greedy(std::int32_t token) override {
        validate_decode();
        if (!decode_greedy_graph_) {
            decode_greedy_graph_ = std::make_unique<ForwardGraph>(
                tensors_, hparams_, cache_, decode_scheduler_.get(), capacity_, 1,
                OutputKind::GreedyToken, InputKind::Decode
            );
        }
        const std::span<const std::int32_t> tokens(&token, 1);
        const std::int32_t output = decode_greedy_graph_->compute_greedy_token(tokens, position_);
        ++position_;
        return output;
    }

    ExecutionBufferMetrics execution_buffers() const override {
        return ExecutionBufferMetrics{
            .kv_cache_bytes = cache_buffer_ == nullptr
                ? 0U
                : ggml_backend_buffer_get_size(cache_buffer_.get()),
            .prefill_scratch_bytes = scheduler_buffer_bytes(prefill_scheduler_.get()),
            .decode_scratch_bytes = scheduler_buffer_bytes(decode_scheduler_.get()),
            .prefill_host_input_bytes = prefill_greedy_graph_ == nullptr
                ? 0U
                : prefill_greedy_graph_->host_input_bytes(),
        };
    }

private:
    void validate_prefill(std::span<const std::int32_t> tokens) const {
        if (position_ != 0 || tokens.empty() || tokens.size() > capacity_) {
            throw std::invalid_argument("native Qwen2 prefill is outside session capacity");
        }
    }

    void validate_decode() const {
        if (position_ == 0 || position_ >= capacity_) {
            throw std::invalid_argument("native Qwen2 decode is outside session capacity");
        }
    }

    void allocate_cache() {
        const std::size_t tensor_count = 2U * hparams_.public_info.layer_count;
        cache_context_.reset(ggml_init(ggml_init_params{
            .mem_size = tensor_count * ggml_tensor_overhead() + 1024U * 1024U,
            .mem_buffer = nullptr,
            .no_alloc = true,
        }));
        if (!cache_context_) {
            throw std::runtime_error("could not allocate native Qwen2 cache metadata");
        }
        const std::int64_t head_length =
            hparams_.public_info.embedding_length / hparams_.head_count;
        const std::int64_t kv_length = head_length * hparams_.kv_head_count;
        cache_.reserve(hparams_.public_info.layer_count);
        for (std::uint32_t layer = 0; layer < hparams_.public_info.layer_count; ++layer) {
            ggml_tensor * key = ggml_new_tensor_2d(
                cache_context_.get(), GGML_TYPE_F16,
                kv_length, static_cast<std::int64_t>(capacity_)
            );
            ggml_tensor * value = ggml_new_tensor_3d(
                cache_context_.get(), GGML_TYPE_F16,
                static_cast<std::int64_t>(capacity_), head_length,
                hparams_.kv_head_count
            );
            ggml_set_name(key, ("native.k." + std::to_string(layer)).c_str());
            ggml_set_name(value, ("native.v." + std::to_string(layer)).c_str());
            cache_.push_back({.key = key, .value = value});
        }
        cache_buffer_.reset(
            ggml_backend_alloc_ctx_tensors(cache_context_.get(), tensors_.backend())
        );
        if (!cache_buffer_) {
            throw std::runtime_error("could not allocate native Qwen2 KV cache");
        }
        ggml_backend_buffer_clear(cache_buffer_.get(), 0);
    }

    ggml_backend_sched_ptr create_scheduler() {
        std::span<ggml_backend_t> backends = tensors_.scheduler_backends();
        ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(
            backends.data(), nullptr, backends.size(), kGraphSize, false, true
        ));
        if (!scheduler) {
            throw std::runtime_error("could not create native Qwen2 scheduler");
        }
        return scheduler;
    }

    static std::uint64_t scheduler_buffer_bytes(ggml_backend_sched_t scheduler) {
        std::uint64_t bytes = 0;
        const int backend_count = ggml_backend_sched_get_n_backends(scheduler);
        for (int index = 0; index < backend_count; ++index) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(scheduler, index);
            bytes += ggml_backend_sched_get_buffer_size(scheduler, backend);
        }
        return bytes;
    }

    TensorStore& tensors_;
    Qwen2HParams hparams_;
    std::size_t capacity_;
    std::size_t position_ = 0;
    ggml_context_ptr cache_context_;
    ggml_backend_buffer_ptr cache_buffer_;
    ggml_backend_sched_ptr prefill_scheduler_;
    ggml_backend_sched_ptr decode_scheduler_;
    std::vector<LayerCache> cache_;
    std::unique_ptr<ForwardGraph> prefill_greedy_graph_;
    std::size_t prefill_token_count_ = 0;
    std::unique_ptr<ForwardGraph> decode_greedy_graph_;
};

class Qwen2Architecture final : public ModelArchitecture {
public:
    Qwen2Architecture(const gguf_context * metadata, const TensorStore& tensors)
        : hparams_(load_qwen2_hparams(metadata)) {
        validate_qwen2_tensors(tensors, hparams_);
    }

    const ModelInfo& model_info() const override { return hparams_.public_info; }

    std::unique_ptr<InferenceSession> create_session(
        TensorStore& tensors,
        std::size_t capacity
    ) const override {
        return std::make_unique<Qwen2Session>(tensors, hparams_, capacity);
    }

private:
    Qwen2HParams hparams_;
};

std::unique_ptr<ModelArchitecture> create_qwen2_architecture(
    const gguf_context * metadata,
    const TensorStore& tensors
) {
    return std::make_unique<Qwen2Architecture>(metadata, tensors);
}

} // namespace jetsonfabric::native
