#include "jetsonfabric/native_inference.hpp"

#include "model_architecture.hpp"
#include "qwen2_metadata.hpp"
#include "tensor_store.hpp"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

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
    ForwardGraph(TensorStore& weights, const Qwen2HParams& params, std::size_t token_count)
        : weights_(weights), params_(params), token_count_(token_count) {
        if (token_count == 0 || token_count > params.public_info.context_length) {
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
    }

    std::vector<float> compute(std::span<const std::int32_t> tokens) {
        ggml_backend_t backend = weights_.backend();
        ggml_backend_t backends[] = {backend};
        ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(
            backends, nullptr, 1, kGraphSize, false, true
        ));
        if (!scheduler || !ggml_backend_sched_alloc_graph(scheduler.get(), graph_)) {
            throw std::runtime_error("could not allocate native Qwen2 compute graph");
        }
        set_inputs(tokens);
        const ggml_status status = ggml_backend_sched_graph_compute(scheduler.get(), graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("native Qwen2 graph computation failed");
        }
        if (logits_->ne[0] <= 0) {
            throw std::runtime_error("native Qwen2 logits have invalid vocabulary dimension");
        }
        std::vector<float> output(static_cast<std::size_t>(logits_->ne[0]));
        ggml_backend_tensor_get(logits_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
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

    ggml_tensor * attention(ggml_tensor * input, std::uint32_t layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const ModelInfo& info = params_.public_info;
        const std::int64_t tokens = static_cast<std::int64_t>(token_count_);
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
        key = ggml_permute(context_.get(), key, 0, 2, 1, 3);
        ggml_tensor * scores = multiply(key, query, "attention scores");
        scores = ggml_soft_max_ext(
            context_.get(), scores, mask_, 1.0F / std::sqrt(static_cast<float>(head_length)), 0.0F
        );

        value = ggml_permute(context_.get(), value, 1, 2, 0, 3);
        value = ggml_cont(context_.get(), value);
        ggml_tensor * attended = multiply(value, scores, "attention values");
        attended = ggml_permute(context_.get(), attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(
            context_.get(), attended, info.embedding_length, tokens
        );
        if (attended->ne[0] != info.embedding_length || value->ne[0] != tokens ||
            value->ne[1] != head_length || key->ne[0] != head_length ||
            key->ne[1] != tokens || key->ne[2] != params_.kv_head_count || kv_length <= 0) {
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
        input_tokens_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32, static_cast<std::int64_t>(token_count_)
        );
        positions_ = ggml_new_tensor_1d(
            context_.get(), GGML_TYPE_I32, static_cast<std::int64_t>(token_count_)
        );
        mask_ = ggml_new_tensor_2d(
            context_.get(), GGML_TYPE_F32,
            static_cast<std::int64_t>(token_count_),
            static_cast<std::int64_t>(token_count_)
        );
        output_index_ = ggml_new_tensor_1d(context_.get(), GGML_TYPE_I32, 1);
        ggml_set_input(input_tokens_);
        ggml_set_input(positions_);
        ggml_set_input(mask_);
        ggml_set_input(output_index_);

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
        logits_ = linear("output.weight", hidden);
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(context_.get(), kGraphSize, false);
        ggml_build_forward_expand(graph_, logits_);
    }

    void set_inputs(std::span<const std::int32_t> tokens) {
        if (tokens.size() != token_count_) {
            throw std::invalid_argument("native Qwen2 graph token count changed");
        }
        std::vector<std::int32_t> positions(token_count_);
        std::vector<float> mask(token_count_ * token_count_);
        for (std::size_t query = 0; query < token_count_; ++query) {
            positions[query] = static_cast<std::int32_t>(query);
            for (std::size_t key = 0; key < token_count_; ++key) {
                mask[query * token_count_ + key] = key > query
                    ? -std::numeric_limits<float>::infinity()
                    : 0.0F;
            }
        }
        const std::int32_t output_index = static_cast<std::int32_t>(token_count_ - 1);
        ggml_backend_tensor_set(input_tokens_, tokens.data(), 0, tokens.size_bytes());
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_tensor_set(output_index_, &output_index, 0, sizeof(output_index));
    }

    TensorStore& weights_;
    const Qwen2HParams& params_;
    std::size_t token_count_;
    ggml_context_ptr context_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_tokens_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_index_ = nullptr;
    ggml_tensor * logits_ = nullptr;
};

} // namespace

class Qwen2Architecture final : public ModelArchitecture {
public:
    Qwen2Architecture(const gguf_context * metadata, const TensorStore& tensors)
        : hparams_(load_qwen2_hparams(metadata)) {
        validate_qwen2_tensors(tensors, hparams_);
    }

    const ModelInfo& model_info() const override { return hparams_.public_info; }

    std::vector<float> logits(
        TensorStore& tensors,
        std::span<const std::int32_t> tokens
    ) const override {
        return ForwardGraph(tensors, hparams_, tokens.size()).compute(tokens);
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
