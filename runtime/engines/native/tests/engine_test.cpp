#include "jetsonfabric/engine.h"

#include "format.h"
#include "sha256.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct SegmentFixture {
    std::uint32_t kind;
    std::int32_t layer;
    std::string path;
    std::string tensor_name;
    std::vector<std::uint8_t> data;
    std::uint64_t file_size = 0;
    std::array<std::uint8_t, 32> sha256{};
};

jf_model * open_model(const fs::path& path, std::uint32_t start, std::uint32_t end);

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
}

void append_i32(std::vector<std::uint8_t>& output, std::int32_t value) {
    append_u32(output, static_cast<std::uint32_t>(value));
}

void append_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
}

void pad_to(std::vector<std::uint8_t>& output, std::size_t alignment) {
    while (output.size() % alignment != 0) {
        output.push_back(0);
    }
}

void write_file(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    expect(output.good(), "could not create fixture file " + path.string());
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    expect(output.good(), "could not write fixture file " + path.string());
}

std::uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') return static_cast<std::uint8_t>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<std::uint8_t>(value - 'a' + 10);
    throw std::runtime_error("golden package contains invalid hex");
}

void decode_hex_file(const fs::path& source, const fs::path& destination) {
    std::ifstream input(source);
    expect(input.good(), "could not open golden fixture " + source.string());
    std::string hex;
    char value = 0;
    while (input.get(value)) {
        if (!std::isspace(static_cast<unsigned char>(value))) hex.push_back(value);
    }
    expect(hex.size() % 2 == 0, "golden fixture has an incomplete byte");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t index = 0; index < hex.size(); index += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            (hex_nibble(hex[index]) << 4U) | hex_nibble(hex[index + 1])
        ));
    }
    write_file(destination, bytes);
}

void test_golden_package(const fs::path& directory) {
    static const std::array<const char *, 6> files = {
        "manifest.jfm", "metadata.gguf", "input.jfs",
        "layer-00000.jfs", "layer-00001.jfs", "output.jfs",
    };
    fs::create_directories(directory);
    const fs::path fixture = fs::path(JF_NATIVE_TEST_SOURCE_DIR) /
        "fixtures" / "jfm-v2-golden";
    for (const char * file : files) {
        decode_hex_file(fixture / (std::string(file) + ".hex"), directory / file);
    }

    jf_model * first = open_model(directory, 0, 1);
    expect(jf_model_get_stats(first).selected_tensor_count == 3, "golden first stage changed");
    jf_model_close(first);
    jf_model * last = open_model(directory, 1, 2);
    expect(jf_model_get_stats(last).selected_tensor_count == 4, "golden last stage changed");
    jf_model_close(last);
}

void overwrite_u32(const fs::path& path, std::uint64_t offset, std::uint32_t value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    expect(file.good(), "could not open fixture for mutation " + path.string());
    file.seekp(static_cast<std::streamoff>(offset));
    for (unsigned index = 0; index < 4; ++index) {
        file.put(static_cast<char>(value >> (index * 8U)));
    }
    expect(file.good(), "could not mutate fixture " + path.string());
}

void overwrite_u64(const fs::path& path, std::uint64_t offset, std::uint64_t value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    expect(file.good(), "could not open fixture for mutation " + path.string());
    file.seekp(static_cast<std::streamoff>(offset));
    for (unsigned index = 0; index < 8; ++index) {
        file.put(static_cast<char>(value >> (index * 8U)));
    }
    expect(file.good(), "could not mutate fixture " + path.string());
}

void write_segment(const fs::path& directory, SegmentFixture& segment) {
    if (segment.kind == JF_SEGMENT_METADATA) {
        segment.file_size = segment.data.size();
        jf_sha256_buffer(segment.data.data(), segment.data.size(), segment.sha256.data());
        write_file(directory / segment.path, segment.data);
        return;
    }
    const std::size_t index_bytes =
        ((JF_TENSOR_RECORD_SIZE + segment.tensor_name.size() + 7U) / 8U) * 8U;
    const std::size_t data_offset =
        ((JF_SEGMENT_HEADER_SIZE + index_bytes + JF_FORMAT_ALIGNMENT - 1U) /
         JF_FORMAT_ALIGNMENT) * JF_FORMAT_ALIGNMENT;

    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), std::begin(JF_SEGMENT_MAGIC), std::end(JF_SEGMENT_MAGIC));
    append_u32(bytes, JF_MODEL_FORMAT_VERSION);
    append_u32(bytes, 1);
    append_u64(bytes, index_bytes);
    append_u64(bytes, segment.data.size());
    bytes.resize(JF_SEGMENT_HEADER_SIZE, 0);

    append_u32(bytes, JF_TENSOR_I8);
    append_u32(bytes, 1); // rank
    append_i32(bytes, segment.layer);
    append_u32(bytes, 0);
    append_u64(bytes, segment.data.size());
    append_u64(bytes, 1);
    append_u64(bytes, 1);
    append_u64(bytes, 1);
    append_u64(bytes, data_offset);
    append_u64(bytes, segment.data.size());
    append_u32(bytes, static_cast<std::uint32_t>(segment.tensor_name.size()));
    append_u32(bytes, 1); // one element per storage block
    append_u32(bytes, 1); // one byte per storage block
    append_u32(bytes, 0);
    bytes.insert(bytes.end(), segment.tensor_name.begin(), segment.tensor_name.end());
    pad_to(bytes, 8);
    bytes.resize(data_offset, 0);
    bytes.insert(bytes.end(), segment.data.begin(), segment.data.end());

    segment.file_size = bytes.size();
    jf_sha256_buffer(bytes.data(), bytes.size(), segment.sha256.data());
    write_file(directory / segment.path, bytes);
}

void write_manifest(
    const fs::path& directory,
    const std::vector<SegmentFixture>& segments,
    std::string_view path_override = {}
) {
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), std::begin(JF_MANIFEST_MAGIC), std::end(JF_MANIFEST_MAGIC));
    append_u32(bytes, JF_MODEL_FORMAT_VERSION);
    append_u32(bytes, 2); // layer count
    append_u64(bytes, 1); // source size
    for (std::uint8_t value = 0; value < 32; ++value) {
        bytes.push_back(value);
    }
    append_u32(bytes, static_cast<std::uint32_t>(segments.size()));
    append_u32(bytes, 0);
    expect(bytes.size() == JF_MANIFEST_HEADER_SIZE, "fixture manifest header size changed");

    for (std::size_t index = 0; index < segments.size(); ++index) {
        const SegmentFixture& segment = segments[index];
        const std::string_view path = index == 0 && !path_override.empty()
            ? path_override
            : std::string_view(segment.path);
        append_u32(bytes, segment.kind);
        append_i32(bytes, segment.layer);
        const bool metadata = segment.kind == JF_SEGMENT_METADATA;
        append_u64(bytes, metadata ? 0 : 1);
        append_u64(bytes, metadata ? 0 : segment.data.size());
        append_u64(bytes, segment.file_size);
        bytes.insert(bytes.end(), segment.sha256.begin(), segment.sha256.end());
        append_u32(bytes, static_cast<std::uint32_t>(path.size()));
        append_u32(bytes, 0);
        expect(bytes.size() >= JF_MANIFEST_SEGMENT_SIZE, "fixture segment record is short");
        bytes.insert(bytes.end(), path.begin(), path.end());
    }
    write_file(directory / "manifest.jfm", bytes);
}

std::vector<SegmentFixture> make_package(const fs::path& directory) {
    fs::create_directories(directory);
    std::vector<SegmentFixture> segments = {
        {JF_SEGMENT_METADATA, -1, "metadata.gguf", "", {'G', 'G', 'U', 'F', 3, 0, 0, 0}},
        {JF_SEGMENT_SHARED, -1, "shared.jfs", "shared.scale", {1, 2}},
        {JF_SEGMENT_INPUT, -1, "input.jfs", "token_embd.weight", {3, 4, 5}},
        {JF_SEGMENT_LAYER, 0, "layer-00000.jfs", "blk.0.weight", {6, 7, 8, 9}},
        {JF_SEGMENT_LAYER, 1, "layer-00001.jfs", "blk.1.weight", {10, 11, 12, 13, 14}},
        {JF_SEGMENT_OUTPUT, -1, "output.jfs", "output.weight", {15, 16, 17, 18, 19, 20}},
    };
    for (SegmentFixture& segment : segments) {
        write_segment(directory, segment);
    }
    write_manifest(directory, segments);
    return segments;
}

jf_model * open_model(const fs::path& path, std::uint32_t start, std::uint32_t end) {
    jf_model * model = nullptr;
    const jf_stage_plan plan = {
        .layer_start = start,
        .layer_end = end,
        .verify_hashes = 1,
        .evict_before_open = 0,
    };
    const jf_status status = jf_model_open(path.c_str(), &plan, &model);
    expect(status.code == JF_STATUS_OK, "open model failed: " + std::string(status.message));
    return model;
}

void test_stage_residency(const fs::path& package) {
    jf_model * first = open_model(package, 0, 1);
    const jf_model_stats first_stats = jf_model_get_stats(first);
    expect(first_stats.layer_start == 0 && first_stats.layer_end == 1, "wrong first stage range");
    expect(first_stats.layer_count == 2, "wrong model layer count");
    expect(first_stats.selected_weight_bytes == 9, "first stage selected the wrong weights");
    expect(first_stats.total_weight_bytes == 20, "wrong total weight bytes");
    expect(first_stats.selected_tensor_count == 3, "first stage selected the wrong tensors");
    expect(first_stats.total_tensor_count == 5, "wrong total tensor count");

    jf_tensor_view tensor{};
    expect(
        jf_model_find_tensor(first, "blk.0.weight", &tensor).code == JF_STATUS_OK,
        "first layer tensor is missing"
    );
    expect(tensor.layer == 0 && tensor.size == 4, "first layer tensor metadata is wrong");
    expect(
        tensor.type == JF_TENSOR_I8 &&
        tensor.storage_block_elements == 1 && tensor.storage_block_bytes == 1,
        "first layer storage contract is wrong"
    );
    expect(
        jf_model_find_tensor(first, "output.weight", &tensor).code == JF_STATUS_NOT_FOUND,
        "first stage incorrectly mapped the output"
    );
    std::uint64_t checksum = 0;
    expect(jf_model_prefetch(first, &checksum).code == JF_STATUS_OK, "prefetch failed");
    expect(checksum != 0, "prefetch did not produce a checksum");
    const void * metadata = nullptr;
    std::size_t metadata_size = 0;
    expect(
        jf_model_get_gguf_metadata(first, &metadata, &metadata_size).code == JF_STATUS_OK &&
        metadata_size == 8 && std::memcmp(metadata, "GGUF", 4) == 0,
        "GGUF metadata was not preserved"
    );
    std::array<std::uint8_t, 32> source_sha{};
    expect(
        jf_model_get_source_sha256(first, source_sha.data()).code == JF_STATUS_OK &&
        source_sha[0] == 0 && source_sha[31] == 31,
        "source identity was not exposed"
    );
    jf_model_close(first);

    jf_model * last = open_model(package, 1, 2);
    const jf_model_stats last_stats = jf_model_get_stats(last);
    expect(last_stats.selected_weight_bytes == 13, "last stage selected the wrong weights");
    expect(last_stats.selected_tensor_count == 3, "last stage selected the wrong tensors");
    expect(
        jf_model_find_tensor(last, "token_embd.weight", &tensor).code == JF_STATUS_NOT_FOUND,
        "last stage incorrectly mapped the input embedding"
    );
    expect(
        jf_model_find_tensor(last, "output.weight", &tensor).code == JF_STATUS_OK,
        "last stage output tensor is missing"
    );
    jf_model_close(last);

    jf_model * full = open_model(package, 0, 2);
    const jf_model_stats full_stats = jf_model_get_stats(full);
    expect(full_stats.selected_weight_bytes == 20, "full model did not select every weight");
    expect(full_stats.selected_tensor_count == 5, "full model did not select every tensor");
    jf_model_close(full);
}

void test_rejects_invalid_inputs(const fs::path& root, const std::vector<SegmentFixture>& segments) {
    jf_model * model = nullptr;
    const jf_stage_plan invalid_range = {
        .layer_start = 1,
        .layer_end = 1,
        .verify_hashes = 0,
        .evict_before_open = 0,
    };
    expect(
        jf_model_open(root.c_str(), &invalid_range, &model).code == JF_STATUS_INVALID_ARGUMENT,
        "empty layer range was accepted"
    );

    const fs::path unsafe = root.parent_path() / "jf-native-unsafe";
    fs::remove_all(unsafe);
    fs::create_directories(unsafe);
    write_manifest(unsafe, segments, "../escape.jfs");
    const jf_stage_plan valid_range = {
        .layer_start = 0,
        .layer_end = 1,
        .verify_hashes = 0,
        .evict_before_open = 0,
    };
    expect(
        jf_model_open(unsafe.c_str(), &valid_range, &model).code == JF_STATUS_FORMAT_ERROR,
        "unsafe segment path was accepted"
    );
    fs::remove_all(unsafe);

    const fs::path corrupt = root.parent_path() / "jf-native-corrupt";
    fs::remove_all(corrupt);
    const std::vector<SegmentFixture> corrupt_segments = make_package(corrupt);
    (void) corrupt_segments;
    std::fstream segment(corrupt / "layer-00000.jfs", std::ios::binary | std::ios::in | std::ios::out);
    segment.seekp(-1, std::ios::end);
    segment.put(static_cast<char>(0xff));
    segment.close();
    const jf_stage_plan verified_range = {
        .layer_start = 0,
        .layer_end = 1,
        .verify_hashes = 1,
        .evict_before_open = 0,
    };
    expect(
        jf_model_open(corrupt.c_str(), &verified_range, &model).code == JF_STATUS_FORMAT_ERROR,
        "corrupted segment passed checksum verification"
    );
    fs::remove_all(corrupt);

    const auto expect_format_rejection = [&model, &valid_range](
        const fs::path& path,
        const std::string& message
    ) {
        expect(
            jf_model_open(path.c_str(), &valid_range, &model).code == JF_STATUS_FORMAT_ERROR,
            message
        );
    };

    const fs::path missing_layer = root.parent_path() / "jf-native-missing-layer";
    fs::remove_all(missing_layer);
    std::vector<SegmentFixture> missing_segments = make_package(missing_layer);
    missing_segments.erase(missing_segments.begin() + 4); // layer 1
    write_manifest(missing_layer, missing_segments);
    expect_format_rejection(missing_layer, "manifest with a missing layer was accepted");
    fs::remove_all(missing_layer);

    const fs::path duplicate_path = root.parent_path() / "jf-native-duplicate-path";
    fs::remove_all(duplicate_path);
    std::vector<SegmentFixture> duplicate_paths = make_package(duplicate_path);
    duplicate_paths[2].path = duplicate_paths[1].path;
    write_manifest(duplicate_path, duplicate_paths);
    expect_format_rejection(duplicate_path, "manifest with duplicate paths was accepted");
    fs::remove_all(duplicate_path);

    const fs::path wrong_layer = root.parent_path() / "jf-native-wrong-layer";
    fs::remove_all(wrong_layer);
    (void) make_package(wrong_layer);
    overwrite_u32(
        wrong_layer / "layer-00000.jfs",
        JF_SEGMENT_HEADER_SIZE + 8,
        1
    );
    expect_format_rejection(wrong_layer, "tensor in the wrong layer segment was accepted");
    fs::remove_all(wrong_layer);

    const fs::path bad_offset = root.parent_path() / "jf-native-bad-offset";
    fs::remove_all(bad_offset);

    const fs::path bad_layout = root.parent_path() / "jf-native-bad-layout";
    fs::remove_all(bad_layout);

    const fs::path reserved_field = root.parent_path() / "jf-native-reserved-field";
    fs::remove_all(reserved_field);
    (void) make_package(reserved_field);
    overwrite_u32(reserved_field / "manifest.jfm", 60, 1);
    expect_format_rejection(reserved_field, "nonzero manifest reserved field was accepted");
    fs::remove_all(reserved_field);
    (void) make_package(bad_layout);
    overwrite_u32(
        bad_layout / "layer-00000.jfs",
        JF_SEGMENT_HEADER_SIZE,
        JF_TENSOR_Q4_K
    );
    expect_format_rejection(bad_layout, "tensor with forged quantization geometry was accepted");
    fs::remove_all(bad_layout);
    (void) make_package(bad_offset);
    overwrite_u64(
        bad_offset / "layer-00000.jfs",
        JF_SEGMENT_HEADER_SIZE + 48,
        0
    );
    expect_format_rejection(bad_offset, "tensor payload overlapping the index was accepted");
    fs::remove_all(bad_offset);

    const fs::path duplicate_name = root.parent_path() / "jf-native-duplicate-name";
    fs::remove_all(duplicate_name);
    std::vector<SegmentFixture> duplicate_names = make_package(duplicate_name);
    duplicate_names[2].tensor_name = duplicate_names[1].tensor_name;
    write_segment(duplicate_name, duplicate_names[2]);
    write_manifest(duplicate_name, duplicate_names);
    expect_format_rejection(duplicate_name, "duplicate selected tensor names were accepted");
    fs::remove_all(duplicate_name);
}

void test_sha256() {
    const std::string input = "abc";
    std::array<std::uint8_t, 32> digest{};
    jf_sha256_buffer(input.data(), input.size(), digest.data());
    const std::array<std::uint8_t, 32> expected = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    expect(digest == expected, "SHA-256 implementation failed the abc test vector");
}

void test_tensor_layout_contract() {
    for (std::uint32_t type = JF_TENSOR_F32; type <= JF_TENSOR_Q2_0; ++type) {
        std::uint32_t elements = 0;
        std::uint32_t bytes = 0;
        expect(
            jf_tensor_type_layout(type, &elements, &bytes).code == JF_STATUS_OK &&
            elements > 0 && bytes > 0,
            "stable tensor type is missing canonical storage geometry"
        );
    }
    std::uint32_t elements = 0;
    std::uint32_t bytes = 0;
    expect(
        jf_tensor_type_layout(JF_TENSOR_Q4_K, &elements, &bytes).code == JF_STATUS_OK &&
        elements == 256 && bytes == 144,
        "Q4_K storage geometry changed"
    );
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "jf-native-engine-test";
    const fs::path golden = root.parent_path() / "jf-native-golden";
    try {
        fs::remove_all(root);
        fs::remove_all(golden);
        const std::vector<SegmentFixture> segments = make_package(root);
        test_golden_package(golden);
        test_sha256();
        test_tensor_layout_contract();
        test_stage_residency(root);
        test_rejects_invalid_inputs(root, segments);
        fs::remove_all(root);
        fs::remove_all(golden);
        std::cout << "native engine tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        fs::remove_all(golden);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
