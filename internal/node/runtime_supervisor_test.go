package node

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestWaitForRuntimeHealthAttestsWireCompatibility(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"status":"ok",
			"engine":"llama.cpp",
			"mode":"pipeline_parallel",
			"stage_transport":"http_binary_v1",
			"activation_encoding":"f16",
			"kv_cache_type":"q8_0",
			"ubatch_size":128
		}`))
	}))
	defer server.Close()

	cfg := Config{
		Engine:                    cluster.EngineLlamaCPP,
		RuntimeMode:               "pipeline_parallel",
		RuntimeStageTransport:     "http_binary_v1",
		RuntimeActivationEncoding: "f16",
		RuntimeKVCacheType:        "q8_0",
		RuntimeUBatchSize:         128,
	}
	if err := waitForRuntimeHealth(
		t.Context(),
		server.URL+"/healthz",
		time.Second,
		cfg,
	); err != nil {
		t.Fatalf("attest runtime health: %v", err)
	}
}

func TestWaitForRuntimeHealthRejectsActivationEncodingMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"status":"ok",
			"engine":"llama.cpp",
			"mode":"pipeline_parallel",
			"stage_transport":"http_binary_v1",
			"activation_encoding":"f32",
			"kv_cache_type":"q8_0",
			"ubatch_size":128
		}`))
	}))
	defer server.Close()

	cfg := Config{
		Engine:                    cluster.EngineLlamaCPP,
		RuntimeMode:               "pipeline_parallel",
		RuntimeStageTransport:     "http_binary_v1",
		RuntimeActivationEncoding: "f16",
		RuntimeKVCacheType:        "q8_0",
		RuntimeUBatchSize:         128,
	}
	err := waitForRuntimeHealth(t.Context(), server.URL+"/healthz", time.Second, cfg)
	if err == nil || !strings.Contains(err.Error(), `activation_encoding="f32", want "f16"`) {
		t.Fatalf("unexpected compatibility result: %v", err)
	}
}

func TestWaitForRuntimeHealthRejectsKVCacheTypeMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"status":"ok",
			"engine":"llama.cpp",
			"mode":"pipeline_parallel",
			"stage_transport":"http_binary_v1",
			"activation_encoding":"f16",
			"kv_cache_type":"f16",
			"ubatch_size":128
		}`))
	}))
	defer server.Close()

	cfg := Config{
		Engine:                    cluster.EngineLlamaCPP,
		RuntimeMode:               "pipeline_parallel",
		RuntimeStageTransport:     "http_binary_v1",
		RuntimeActivationEncoding: "f16",
		RuntimeKVCacheType:        "q8_0",
		RuntimeUBatchSize:         128,
	}
	err := waitForRuntimeHealth(t.Context(), server.URL+"/healthz", time.Second, cfg)
	if err == nil || !strings.Contains(err.Error(), `kv_cache_type="f16", want "q8_0"`) {
		t.Fatalf("unexpected compatibility result: %v", err)
	}
}

func TestWaitForRuntimeHealthRejectsUBatchSizeMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"status":"ok",
			"engine":"llama.cpp",
			"mode":"pipeline_parallel",
			"stage_transport":"http_binary_v1",
			"activation_encoding":"f16",
			"kv_cache_type":"q8_0",
			"ubatch_size":256
		}`))
	}))
	defer server.Close()

	cfg := Config{
		Engine:                    cluster.EngineLlamaCPP,
		RuntimeMode:               "pipeline_parallel",
		RuntimeStageTransport:     "http_binary_v1",
		RuntimeActivationEncoding: "f16",
		RuntimeKVCacheType:        "q8_0",
		RuntimeUBatchSize:         128,
	}
	err := waitForRuntimeHealth(t.Context(), server.URL+"/healthz", time.Second, cfg)
	if err == nil || !strings.Contains(err.Error(), "ubatch_size=256, want 128") {
		t.Fatalf("unexpected compatibility result: %v", err)
	}
}
