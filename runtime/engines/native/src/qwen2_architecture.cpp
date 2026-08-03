#include "jetsonfabric/native_inference.hpp"

#include "model_architecture.hpp"
#include "qwen2_metadata.hpp"
#include "tensor_store.hpp"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
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
constexpr std::size_t kFlashAttentionKvAlignment = 256;
constexpr std::int64_t kMinFusedSwiGLUElements = 10'000;

constexpr std::size_t physical_cache_capacity(
    std::size_t logical_capacity,
    std::size_t model_context_length,
    AttentionKernel decode_attention_kernel
) {
    if (decode_attention_kernel != AttentionKernel::Flash) {
        return logical_capacity;
    }
    const std::size_t aligned =
        (logical_capacity + kFlashAttentionKvAlignment - 1U) /
        kFlashAttentionKvAlignment * kFlashAttentionKvAlignment;
    return std::min(aligned, model_context_length);
}

static_assert(physical_cache_capacity(2079, 32768, AttentionKernel::Flash) == 2304);
static_assert(physical_cache_capacity(2079, 32768, AttentionKernel::Unfused) == 2079);
static_assert(physical_cache_capacity(32760, 32768, AttentionKernel::Flash) == 32768);

struct LayerCache {
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
};

enum class OutputKind { Activations, Logits, GreedyToken };
enum class InputKind { Prefill, Decode };

struct GraphTimings {
    bool attention_backend_verified = false;
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

void validate_qwen2_tensors(
    const TensorStore& tensors,
    const Qwen2HParams& params,
    LayerRange layers
) {
    const std::int64_t embedding = params.public_info.embedding_length;
    const std::int64_t head_length = embedding / params.head_count;
    const std::int64_t kv_length = head_length * params.kv_head_count;
    const std::int64_t feed_forward = params.feed_forward_length;
    if (layers.start >= layers.end || layers.end > params.public_info.layer_count) {
        throw std::runtime_error("native Qwen2 layer range is invalid");
    }
    const std::int64_t vocabulary = tensors.vocabulary_size();
    if (layers.is_first()) {
        require_shape(tensors, "token_embd.weight", {embedding, vocabulary});
    }
    if (layers.is_last(params.public_info.layer_count)) {
        require_shape(tensors, "output_norm.weight", {embedding});
        require_shape(tensors, "output.weight", {embedding, vocabulary});
        if (tensors.find("output.bias") != nullptr) {
            require_shape(tensors, "output.bias", {vocabulary});
        }
    }
    for (std::uint32_t layer = layers.start; layer < layers.end; ++layer) {
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
        InputKind input_kind,
        AttentionKernel attention_kernel,
        LayerRange layers
    ) : weights_(weights), params_(params), cache_(cache), scheduler_(scheduler),
        attention_length_(attention_length), token_count_(token_count),
        output_kind_(output_kind), input_kind_(input_kind),
        attention_kernel_(attention_kernel), layers_(layers) {
        if (token_count == 0 || attention_length < token_count ||
            attention_length > params.public_info.context_length ||
            layers.start >= layers.end || layers.end > params.public_info.layer_count ||
            cache.size() != layers.end - layers.start || scheduler == nullptr) {
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

    std::vector<float> compute_activations(
        std::span<const std::int32_t> tokens,
        std::size_t position
    ) {
        return compute_activations(tokens, {}, position);
    }

    std::vector<float> compute_activations(
        std::span<const float> activations,
        std::size_t position
    ) {
        return compute_activations({}, activations, position);
    }

    std::int32_t compute_greedy_token(
        std::span<const float> activations,
        std::size_t position,
        GraphTimings * timings = nullptr
    ) {
        if (output_kind_ != OutputKind::GreedyToken) {
            throw std::logic_error("native Qwen2 graph does not produce a greedy token");
        }
        compute({}, activations, position, timings);
        std::int32_t token = -1;
        ggml_backend_tensor_get(greedy_token_, &token, 0, sizeof(token));
        return token;
    }

    std::uint64_t host_input_bytes() const {
        return fixed_positions_.size() * sizeof(std::int32_t) +
            fixed_unfused_attention_mask_.size() * sizeof(float) +
            fixed_flash_attention_mask_.size() * sizeof(ggml_fp16_t) +
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
        compute(tokens, {}, position, timings);
    }

    void compute(
        std::span<const std::int32_t> tokens,
        std::span<const float> activations,
        std::size_t position,
        GraphTimings * timings = nullptr
    ) {
        if (!allocated_) {
            const auto allocation_start = std::chrono::steady_clock::now();
            ggml_backend_sched_reset(scheduler_);
            if (!ggml_backend_sched_alloc_graph(scheduler_, graph_)) {
                throw std::runtime_error("could not allocate native Qwen2 compute graph");
            }
            verify_attention_backend();
            allocated_ = true;
            if (timings != nullptr) {
                timings->allocation_ms = elapsed_ms(allocation_start);
            }
        }
        const auto input_start = std::chrono::steady_clock::now();
        if (timings != nullptr) {
            timings->attention_backend_verified = attention_backend_verified_;
        }
        set_inputs(tokens, activations, position);
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

    std::vector<float> compute_activations(
        std::span<const std::int32_t> tokens,
        std::span<const float> activations,
        std::size_t position
    ) {
        if (output_kind_ != OutputKind::Activations) {
            throw std::logic_error("native Qwen2 graph does not produce activations");
        }
        compute(tokens, activations, position);
        const std::size_t count = token_count_ * params_.public_info.embedding_length;
        std::vector<float> output(count);
        ggml_backend_tensor_get(hidden_output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

    void assign_backends() {
        const ggml_backend_t input_backend = weights_.input_backend();
        if (input_tokens_ != nullptr) {
            ggml_backend_sched_set_tensor_backend(scheduler_, input_tokens_, input_backend);
        }
        if (input_activations_ != nullptr) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_, input_activations_, input_backend
            );
        }
        ggml_backend_sched_set_tensor_backend(scheduler_, positions_, input_backend);
        ggml_backend_sched_set_tensor_backend(scheduler_, mask_, input_backend);
        if (output_kind_ != OutputKind::Activations) {
            ggml_backend_sched_set_tensor_backend(scheduler_, output_index_, input_backend);
        }
        ggml_backend_sched_set_tensor_backend(scheduler_, cache_indices_, input_backend);
        ggml_backend_sched_set_tensor_backend(
            scheduler_, value_cache_indices_, input_backend
        );

        const ggml_backend_t compute_backend = weights_.backend();
        if (hidden_output_ != nullptr) {
            ggml_backend_sched_set_tensor_backend(scheduler_, hidden_output_, compute_backend);
        }
        if (logits_ != nullptr) {
            ggml_backend_sched_set_tensor_backend(scheduler_, logits_, compute_backend);
        }
        if (greedy_token_ != nullptr) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_, greedy_token_, compute_backend
            );
        }
    }

    void verify_attention_backend() {
        if (attention_kernel_ != AttentionKernel::Flash) return;
        if (flash_attention_nodes_.size() != layers_.end - layers_.start) {
            throw std::runtime_error("native Qwen2 flash-attention graph is incomplete");
        }
        for (ggml_tensor * node : flash_attention_nodes_) {
            if (ggml_backend_sched_get_tensor_backend(scheduler_, node) != weights_.backend()) {
                throw std::runtime_error(
                    "native Qwen2 flash attention was not assigned to the compute backend"
                );
            }
        }
        attention_backend_verified_ = true;
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

        const LayerCache& layer_cache = cache_[layer - layers_.start];
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
        ggml_tensor * attended = nullptr;
        const float scale = 1.0F / std::sqrt(static_cast<float>(head_length));
        if (attention_kernel_ == AttentionKernel::Flash) {
            value = ggml_cont(
                context_.get(), ggml_permute(context_.get(), value, 1, 0, 2, 3)
            );
            validate_flash_attention_tensors(query, key, value);
            attended = ggml_flash_attn_ext(
                context_.get(), query, key, value, mask_, scale, 0.0F, 0.0F
            );
            ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
            flash_attention_nodes_.push_back(attended);
            attended = ggml_reshape_2d(
                context_.get(), attended, info.embedding_length, tokens
            );
        } else {
            ggml_tensor * scores = multiply(key, query, "attention scores");
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores = ggml_soft_max_ext(
                context_.get(), scores, mask_, scale, 0.0F
            );
            attended = multiply(value, scores, "attention values");
            attended = ggml_permute(context_.get(), attended, 0, 2, 1, 3);
            attended = ggml_cont_2d(
                context_.get(), attended, info.embedding_length, tokens
            );
        }
        if (attended->ne[0] != info.embedding_length || key->ne[0] != head_length ||
            key->ne[1] != key_count || key->ne[2] != params_.kv_head_count || kv_length <= 0) {
            throw std::runtime_error("native Qwen2 attention graph has inconsistent dimensions");
        }
        return linear(prefix + "attn_output.weight", attended);
    }

    void validate_flash_attention_tensors(
        const ggml_tensor * query,
        const ggml_tensor * key,
        const ggml_tensor * value
    ) const {
        const ModelInfo& info = params_.public_info;
        const std::int64_t head_length = info.embedding_length / params_.head_count;
        const std::int64_t tokens = static_cast<std::int64_t>(token_count_);
        const std::int64_t key_count = static_cast<std::int64_t>(attention_length_);
        const bool valid = params_.kv_head_count > 0 &&
            params_.head_count % params_.kv_head_count == 0 &&
            query->ne[0] == head_length && query->ne[1] == tokens &&
            query->ne[2] == params_.head_count &&
            key->ne[0] == head_length && key->ne[1] == key_count &&
            key->ne[2] == params_.kv_head_count &&
            value->ne[0] == head_length && value->ne[1] == key_count &&
            value->ne[2] == params_.kv_head_count &&
            mask_->type == GGML_TYPE_F16 && mask_->ne[0] == key_count &&
            mask_->ne[1] == tokens && ggml_is_contiguous(value) &&
            ggml_is_contiguous(mask_);
        if (!valid) {
            throw std::runtime_error(
                "native Qwen2 flash-attention tensors violate the GGML contract"
            );
        }
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
        // Small decode vectors are faster as separate kernels on Orin.
        ggml_tensor * activated;
        if (ggml_nelements(gate) >= kMinFusedSwiGLUElements) {
            activated = ggml_swiglu_split(context_.get(), gate, up);
        } else {
            activated = ggml_mul(context_.get(), ggml_silu(context_.get(), gate), up);
        }
        return linear(prefix + "ffn_down.weight", activated);
    }

    void build() {
        const ModelInfo& info = params_.public_info;
        const std::int64_t key_count = static_cast<std::int64_t>(attention_length_);
        if (layers_.is_first()) {
            input_tokens_ = ggml_new_tensor_1d(
                context_.get(), GGML_TYPE_I32,
                static_cast<std::int64_t>(token_count_)
            );
            ggml_set_input(input_tokens_);
        } else {
            input_activations_ = ggml_new_tensor_2d(
                context_.get(), GGML_TYPE_F32, info.embedding_length,
                static_cast<std::int64_t>(token_count_)
            );
            ggml_set_input(input_activations_);
        }
        positions_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32, static_cast<std::int64_t>(token_count_)
        );
        mask_ = ggml_new_tensor_2d(
            context_.get(), attention_kernel_ == AttentionKernel::Flash
                ? GGML_TYPE_F16
                : GGML_TYPE_F32,
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
        ggml_set_input(positions_);
        ggml_set_input(mask_);
        ggml_set_input(output_index_);
        ggml_set_input(cache_indices_);
        ggml_set_input(value_cache_indices_);
        graph_ = ggml_new_graph_custom(context_.get(), kGraphSize, false);

        ggml_tensor * hidden = layers_.is_first()
            ? ggml_get_rows(
                  context_.get(), weights_.require("token_embd.weight"), input_tokens_
              )
            : input_activations_;
        for (std::uint32_t layer = layers_.start; layer < layers_.end; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            ggml_tensor * attention_input = hidden;
            ggml_tensor * normalized = rms_norm(hidden, prefix + "attn_norm.weight");
            hidden = ggml_add(context_.get(), attention(normalized, layer), attention_input);
            ggml_tensor * ffn_input = hidden;
            normalized = rms_norm(hidden, prefix + "ffn_norm.weight");
            hidden = ggml_add(context_.get(), feed_forward(normalized, layer), ffn_input);
        }
        if (output_kind_ == OutputKind::Activations) {
            hidden_output_ = hidden;
            ggml_set_output(hidden_output_);
            ggml_build_forward_expand(graph_, hidden_output_);
            return;
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

    void set_inputs(
        std::span<const std::int32_t> tokens,
        std::span<const float> activations,
        std::size_t position
    ) {
        const std::size_t activation_count =
            token_count_ * params_.public_info.embedding_length;
        const bool input_matches = layers_.is_first()
            ? tokens.size() == token_count_ && activations.empty()
            : tokens.empty() && activations.size() == activation_count;
        if (!input_matches || position + token_count_ > attention_length_) {
            throw std::invalid_argument("native Qwen2 graph token count changed");
        }
        if (input_kind_ == InputKind::Prefill && position != 0) {
            throw std::invalid_argument("native Qwen2 prefill graph must start at position zero");
        }
        if (layers_.is_first()) {
            ggml_backend_tensor_set(input_tokens_, tokens.data(), 0, tokens.size_bytes());
        } else {
            ggml_backend_tensor_set(
                input_activations_, activations.data(), 0, activations.size_bytes()
            );
        }
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
        upload_attention_mask(mask);
        if (output_kind_ != OutputKind::Activations) {
            ggml_backend_tensor_set(output_index_, &output_index, 0, sizeof(output_index));
        }
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
        const std::vector<float> mask = build_attention_mask(0);
        if (attention_kernel_ == AttentionKernel::Flash) {
            fixed_flash_attention_mask_ = to_fp16(mask);
        } else {
            fixed_unfused_attention_mask_ = mask;
        }
        fixed_value_cache_indices_ = build_value_cache_indices(0);
        fixed_output_index_ = static_cast<std::int32_t>(token_count_ - 1);
    }

    void upload_fixed_inputs() {
        ggml_backend_tensor_set(
            positions_, fixed_positions_.data(), 0,
            fixed_positions_.size() * sizeof(std::int32_t)
        );
        if (attention_kernel_ == AttentionKernel::Flash) {
            ggml_backend_tensor_set(
                mask_, fixed_flash_attention_mask_.data(), 0,
                fixed_flash_attention_mask_.size() * sizeof(ggml_fp16_t)
            );
        } else {
            ggml_backend_tensor_set(
                mask_, fixed_unfused_attention_mask_.data(), 0,
                fixed_unfused_attention_mask_.size() * sizeof(float)
            );
        }
        if (output_kind_ != OutputKind::Activations) {
            ggml_backend_tensor_set(
                output_index_, &fixed_output_index_, 0, sizeof(fixed_output_index_)
            );
        }
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

    static std::vector<ggml_fp16_t> to_fp16(std::span<const float> source) {
        std::vector<ggml_fp16_t> converted(source.size());
        for (std::size_t index = 0; index < source.size(); ++index) {
            converted[index] = ggml_fp32_to_fp16(source[index]);
        }
        return converted;
    }

    void upload_attention_mask(std::span<const float> mask) {
        if (attention_kernel_ == AttentionKernel::Flash) {
            const std::vector<ggml_fp16_t> converted = to_fp16(mask);
            ggml_backend_tensor_set(
                mask_, converted.data(), 0,
                converted.size() * sizeof(ggml_fp16_t)
            );
            return;
        }
        ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size_bytes());
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
    AttentionKernel attention_kernel_;
    LayerRange layers_;
    bool allocated_ = false;
    bool attention_backend_verified_ = false;
    ggml_context_ptr context_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_tokens_ = nullptr;
    ggml_tensor * input_activations_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_index_ = nullptr;
    ggml_tensor * cache_indices_ = nullptr;
    ggml_tensor * value_cache_indices_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_tensor * greedy_token_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    std::vector<ggml_tensor *> flash_attention_nodes_;
    std::vector<std::int32_t> fixed_positions_;
    std::vector<float> fixed_unfused_attention_mask_;
    std::vector<ggml_fp16_t> fixed_flash_attention_mask_;
    std::vector<std::int32_t> fixed_value_cache_indices_;
    std::int32_t fixed_output_index_ = 0;
};

} // namespace

class Qwen2Session final : public InferenceSession {
public:
    Qwen2Session(
        TensorStore& tensors,
        Qwen2HParams hparams,
        std::size_t capacity,
        AttentionKernel prefill_attention_kernel,
        AttentionKernel decode_attention_kernel,
        LayerRange layers
    ) : tensors_(tensors), hparams_(std::move(hparams)), capacity_(capacity),
        cache_capacity_(physical_cache_capacity(
            capacity, hparams_.public_info.context_length, decode_attention_kernel
        )),
        prefill_attention_kernel_(prefill_attention_kernel),
        decode_attention_kernel_(decode_attention_kernel), layers_(layers) {
        if (capacity == 0 || capacity > hparams_.public_info.context_length) {
            throw std::invalid_argument("native Qwen2 cache capacity is outside model context");
        }
        allocate_cache();
        prefill_scheduler_ = create_scheduler();
        decode_scheduler_ = create_scheduler();
    }

    std::size_t capacity() const override { return capacity_; }
    std::size_t position() const override { return position_; }
    void reset() override {
        position_ = 0;
        prefill_position_ = 0;
        decode_greedy_graph_.reset();
        stage_prefill_graph_.reset();
        stage_decode_graph_.reset();
    }

    void rollback(std::size_t token_count) override {
        if (token_count == 0 || token_count > position_ - prefill_position_) {
            throw std::invalid_argument("native Qwen2 rollback exceeds the decoded suffix");
        }
        position_ -= token_count;
    }

    std::vector<float> prefill_logits(
        std::span<const std::int32_t> tokens
    ) override {
        require_full_model();
        validate_prefill(tokens);
        ggml_backend_sched_ptr logits_scheduler = create_scheduler();
        std::vector<float> output = ForwardGraph(
            tensors_, hparams_, cache_, logits_scheduler.get(),
            tokens.size(), tokens.size(), OutputKind::Logits,
            InputKind::Prefill, prefill_attention_kernel_, layers_
        ).compute_logits(tokens, 0);
        position_ += tokens.size();
        prefill_position_ = position_;
        return output;
    }

    PrefillResult prefill_greedy(
        std::span<const std::int32_t> tokens
    ) override {
        require_full_model();
        validate_prefill(tokens);
        const bool graph_reused = prefill_greedy_graph_ != nullptr &&
            prefill_token_count_ == tokens.size();
        double graph_build_ms = 0.0;
        if (!graph_reused) {
            const auto build_start = std::chrono::steady_clock::now();
            prefill_greedy_graph_ = std::make_unique<ForwardGraph>(
                tensors_, hparams_, cache_, prefill_scheduler_.get(),
                tokens.size(), tokens.size(), OutputKind::GreedyToken,
                InputKind::Prefill, prefill_attention_kernel_, layers_
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
        prefill_position_ = position_;
        return PrefillResult{
            .token = token,
            .metrics = PrefillMetrics{
                .plan_reused = graph_reused,
                .attention_backend_verified = graph_timings.attention_backend_verified,
                .planning_ms = graph_build_ms,
                .allocation_ms = graph_timings.allocation_ms,
                .host_input_preparation_ms = graph_timings.host_input_preparation_ms,
                .compute_ms = graph_timings.compute_ms,
                .output_read_ms = graph_timings.output_read_ms,
            },
        };
    }

    std::vector<float> decode_logits(std::int32_t token) override {
        require_full_model();
        validate_decode();
        ggml_backend_sched_ptr logits_scheduler = create_scheduler();
        const std::span<const std::int32_t> tokens(&token, 1);
        std::vector<float> output = ForwardGraph(
            tensors_, hparams_, cache_, logits_scheduler.get(), cache_capacity_, 1,
            OutputKind::Logits, InputKind::Decode, decode_attention_kernel_, layers_
        ).compute_logits(tokens, position_);
        ++position_;
        return output;
    }

    std::int32_t decode_greedy(std::int32_t token) override {
        require_full_model();
        validate_decode();
        if (!decode_greedy_graph_) {
            decode_greedy_graph_ = std::make_unique<ForwardGraph>(
                tensors_, hparams_, cache_, decode_scheduler_.get(), cache_capacity_, 1,
                OutputKind::GreedyToken, InputKind::Decode, decode_attention_kernel_, layers_
            );
        }
        const std::span<const std::int32_t> tokens(&token, 1);
        const std::int32_t output = decode_greedy_graph_->compute_greedy_token(tokens, position_);
        ++position_;
        return output;
    }

    StageResult prefill_stage_tokens(
        std::span<const std::int32_t> tokens
    ) override {
        if (!layers_.is_first()) {
            throw std::invalid_argument("only the first native stage accepts tokens");
        }
        validate_prefill(tokens);
        return run_stage_prefill(tokens, {});
    }

    StageResult prefill_stage_activations(
        std::span<const float> activations,
        std::size_t token_count
    ) override {
        if (layers_.is_first()) {
            throw std::invalid_argument("the first native stage does not accept activations");
        }
        validate_stage_activations(activations, token_count);
        validate_prefill_count(token_count);
        return run_stage_prefill({}, activations);
    }

    StageResult decode_stage_token(std::int32_t token) override {
        if (!layers_.is_first()) {
            throw std::invalid_argument("only the first native stage accepts decode tokens");
        }
        validate_decode();
        const std::span<const std::int32_t> tokens(&token, 1);
        return run_stage_decode(tokens, {});
    }

    StageResult decode_stage_activation(
        std::span<const float> activation
    ) override {
        if (layers_.is_first()) {
            throw std::invalid_argument("the first native stage does not accept decode activations");
        }
        validate_stage_activations(activation, 1);
        validate_decode();
        return run_stage_decode({}, activation);
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
    bool final_stage() const noexcept {
        return layers_.is_last(hparams_.public_info.layer_count);
    }

    void require_full_model() const {
        if (!layers_.is_first() || !final_stage()) {
            throw std::logic_error("full-model native inference requires every model layer");
        }
    }

    OutputKind stage_output_kind() const noexcept {
        return final_stage() ? OutputKind::GreedyToken : OutputKind::Activations;
    }

    StageResult run_stage_prefill(
        std::span<const std::int32_t> tokens,
        std::span<const float> activations
    ) {
        const std::size_t token_count = tokens.empty()
            ? activations.size() / hparams_.public_info.embedding_length
            : tokens.size();
        stage_prefill_graph_ = std::make_unique<ForwardGraph>(
            tensors_, hparams_, cache_, prefill_scheduler_.get(),
            token_count, token_count, stage_output_kind(),
            InputKind::Prefill, prefill_attention_kernel_, layers_
        );
        StageResult result;
        if (final_stage()) {
            result.sampled_token = tokens.empty()
                ? stage_prefill_graph_->compute_greedy_token(activations, 0)
                : stage_prefill_graph_->compute_greedy_token(tokens, 0);
        } else {
            result.activations = tokens.empty()
                ? stage_prefill_graph_->compute_activations(activations, 0)
                : stage_prefill_graph_->compute_activations(tokens, 0);
        }
        position_ = token_count;
        prefill_position_ = position_;
        return result;
    }

    StageResult run_stage_decode(
        std::span<const std::int32_t> tokens,
        std::span<const float> activations
    ) {
        if (!stage_decode_graph_) {
            stage_decode_graph_ = std::make_unique<ForwardGraph>(
                tensors_, hparams_, cache_, decode_scheduler_.get(), cache_capacity_, 1,
                stage_output_kind(), InputKind::Decode,
                decode_attention_kernel_, layers_
            );
        }
        StageResult result;
        if (final_stage()) {
            result.sampled_token = tokens.empty()
                ? stage_decode_graph_->compute_greedy_token(activations, position_)
                : stage_decode_graph_->compute_greedy_token(tokens, position_);
        } else {
            result.activations = tokens.empty()
                ? stage_decode_graph_->compute_activations(activations, position_)
                : stage_decode_graph_->compute_activations(tokens, position_);
        }
        ++position_;
        return result;
    }

    void validate_stage_activations(
        std::span<const float> activations,
        std::size_t token_count
    ) const {
        const std::size_t embedding = hparams_.public_info.embedding_length;
        if (token_count == 0 || token_count > std::numeric_limits<std::size_t>::max() / embedding ||
            activations.size() != token_count * embedding) {
            throw std::invalid_argument("native stage activation shape is invalid");
        }
    }

    void validate_prefill_count(std::size_t token_count) const {
        if (position_ != 0 || token_count == 0 || token_count > capacity_) {
            throw std::invalid_argument("native Qwen2 prefill is outside session capacity");
        }
    }

    void validate_prefill(std::span<const std::int32_t> tokens) const {
        validate_prefill_count(tokens.size());
    }

    void validate_decode() const {
        if (position_ == 0 || position_ >= capacity_) {
            throw std::invalid_argument("native Qwen2 decode is outside session capacity");
        }
    }

    void allocate_cache() {
        const std::size_t resident_layer_count = layers_.end - layers_.start;
        const std::size_t tensor_count = 2U * resident_layer_count;
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
        cache_.reserve(resident_layer_count);
        for (std::uint32_t layer = layers_.start; layer < layers_.end; ++layer) {
            ggml_tensor * key = ggml_new_tensor_2d(
                cache_context_.get(), GGML_TYPE_F16,
                kv_length, static_cast<std::int64_t>(cache_capacity_)
            );
            ggml_tensor * value = ggml_new_tensor_3d(
                cache_context_.get(), GGML_TYPE_F16,
                static_cast<std::int64_t>(cache_capacity_), head_length,
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
    std::size_t cache_capacity_;
    AttentionKernel prefill_attention_kernel_;
    AttentionKernel decode_attention_kernel_;
    LayerRange layers_;
    std::size_t position_ = 0;
    std::size_t prefill_position_ = 0;
    ggml_context_ptr cache_context_;
    ggml_backend_buffer_ptr cache_buffer_;
    ggml_backend_sched_ptr prefill_scheduler_;
    ggml_backend_sched_ptr decode_scheduler_;
    std::vector<LayerCache> cache_;
    std::unique_ptr<ForwardGraph> prefill_greedy_graph_;
    std::size_t prefill_token_count_ = 0;
    std::unique_ptr<ForwardGraph> decode_greedy_graph_;
    std::unique_ptr<ForwardGraph> stage_prefill_graph_;
    std::unique_ptr<ForwardGraph> stage_decode_graph_;
};

class Qwen2Architecture final : public ModelArchitecture {
public:
    Qwen2Architecture(const gguf_context * metadata, const TensorStore& tensors)
        : hparams_(load_qwen2_hparams(metadata)) {
        validate_qwen2_tensors(
            tensors,
            hparams_,
            LayerRange{tensors.layer_start(), tensors.layer_end()}
        );
    }

    const ModelInfo& model_info() const override { return hparams_.public_info; }

    AttentionKernel resolve_attention_kernel(
        Backend backend,
        AttentionKernel requested
    ) const override {
        const bool supported = backend == Backend::Cuda && flash_attention_supported();
        if (requested == AttentionKernel::Automatic) {
            return supported ? AttentionKernel::Flash : AttentionKernel::Unfused;
        }
        if (requested == AttentionKernel::Flash && !supported) {
            throw std::invalid_argument(
                backend == Backend::Cuda
                    ? "flash attention does not support this Qwen2 head configuration"
                    : "flash attention requires the CUDA backend"
            );
        }
        return requested;
    }

    std::unique_ptr<InferenceSession> create_session(
        TensorStore& tensors,
        std::size_t capacity,
        AttentionKernel prefill_attention_kernel,
        AttentionKernel decode_attention_kernel,
        LayerRange layers
    ) const override {
        return std::make_unique<Qwen2Session>(
            tensors, hparams_, capacity,
            prefill_attention_kernel, decode_attention_kernel, layers
        );
    }

private:
    bool flash_attention_supported() const {
        if (hparams_.head_count == 0 || hparams_.kv_head_count == 0 ||
            hparams_.public_info.embedding_length % hparams_.head_count != 0 ||
            hparams_.head_count % hparams_.kv_head_count != 0) {
            return false;
        }
        const std::uint32_t head_length =
            hparams_.public_info.embedding_length / hparams_.head_count;
        switch (head_length) {
        case 40:
        case 64:
        case 72:
        case 80:
        case 96:
        case 112:
        case 128:
        case 256:
            return true;
        default:
            return false;
        }
    }

    Qwen2HParams hparams_;
};

std::unique_ptr<ModelArchitecture> create_qwen2_architecture(
    const gguf_context * metadata,
    const TensorStore& tensors
) {
    return std::make_unique<Qwen2Architecture>(metadata, tensors);
}

} // namespace jetsonfabric::native
