package runtimebridge

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/modelartifacts"
)

func TestDeploymentProxyVerifiesArtifactBeforeForwardingLoad(t *testing.T) {
	modelPath := writeDeploymentTestModel(t)
	modelSHA256, err := modelartifacts.ComputeSHA256(modelPath)
	if err != nil {
		t.Fatal(err)
	}

	forwarded := 0
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		forwarded++
		if request.Header.Get(api.HeaderCoordinatorNodeID) != "" || request.Header.Get(api.HeaderClusterToken) != "" {
			t.Fatal("proxy forwarded node-internal authorization headers to the runtime")
		}
		payload, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		var load LoadDeploymentRequest
		if err := json.Unmarshal(payload, &load); err != nil {
			t.Fatal(err)
		}
		if load.Epoch != 7 || load.ModelPath != modelPath || load.ModelSHA256 != modelSHA256 {
			t.Fatalf("unexpected forwarded load identity: %+v", load)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"loaded":true}`))
	}))
	defer runtime.Close()

	proxy, err := NewDeploymentProxy(runtime.URL)
	if err != nil {
		t.Fatal(err)
	}
	request := deploymentLoadProxyRequest(t, modelPath, modelSHA256)
	request.Header.Set(api.HeaderCoordinatorNodeID, "leader")
	request.Header.Set(api.HeaderClusterToken, "secret")
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)

	if response.Code != http.StatusOK || forwarded != 1 {
		t.Fatalf("verified load status=%d forwarded=%d body=%s", response.Code, forwarded, response.Body.String())
	}
}

func TestDeploymentProxyRejectsArtifactMismatchBeforeRuntimeLoad(t *testing.T) {
	modelPath := writeDeploymentTestModel(t)
	forwarded := 0
	runtime := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		forwarded++
	}))
	defer runtime.Close()

	proxy, err := NewDeploymentProxy(runtime.URL)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, deploymentLoadProxyRequest(t, modelPath, string(bytes.Repeat([]byte{'0'}, 64))))

	if response.Code != http.StatusConflict || forwarded != 0 {
		t.Fatalf("mismatched load status=%d forwarded=%d body=%s", response.Code, forwarded, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "error", "model_artifact_mismatch")
}

func TestDeploymentProxyRejectsIncompleteLoadIdentity(t *testing.T) {
	proxy, err := NewDeploymentProxy("http://127.0.0.1:9090")
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(`{"deployment_id":"deployment-a","model_path":"/models/a.gguf","model_sha256":"bad"}`)
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, httptest.NewRequest(http.MethodPost, api.PathRuntimeDeploymentLoad, bytes.NewReader(payload)))

	if response.Code != http.StatusBadRequest {
		t.Fatalf("incomplete load status=%d body=%s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "error", "invalid_load_identity")
}

func TestDeploymentProxyForwardsDrainRoute(t *testing.T) {
	var method, path string
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		method, path = request.Method, request.URL.Path
		_, _ = w.Write([]byte(`{"drained":true}`))
	}))
	defer runtime.Close()
	proxy, err := NewDeploymentProxy(runtime.URL)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, httptest.NewRequest(http.MethodPost, api.PathRuntimeDeploymentDrain, strings.NewReader(`{}`)))
	if response.Code != http.StatusOK || method != http.MethodPost || path != runtimeDeploymentDrainPath {
		t.Fatalf("drain forwarding status=%d method=%q path=%q", response.Code, method, path)
	}
}

func deploymentLoadProxyRequest(t *testing.T, modelPath, modelSHA256 string) *http.Request {
	t.Helper()
	payload, err := json.Marshal(LoadDeploymentRequest{
		DeploymentID: "deployment-a",
		Epoch:        7,
		ModelID:      "model-a",
		ModelSHA256:  modelSHA256,
		ModelPath:    modelPath,
	})
	if err != nil {
		t.Fatal(err)
	}
	return httptest.NewRequest(http.MethodPost, api.PathRuntimeDeploymentLoad, bytes.NewReader(payload))
}

func writeDeploymentTestModel(t *testing.T) string {
	t.Helper()
	path := t.TempDir() + string(os.PathSeparator) + "model.gguf"
	if err := os.WriteFile(path, []byte("deployment artifact"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
