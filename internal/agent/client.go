package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/system"
)

type Client struct {
	controlURL string
	joinToken  string
	nodeID     string
	backends   []cluster.RuntimeBackend
	httpClient *http.Client
}

func NewClient(controlURL string, joinToken string, nodeID string, backends []cluster.RuntimeBackend) *Client {
	return &Client{
		controlURL: strings.TrimSpace(controlURL),
		joinToken:  joinToken,
		nodeID:     nodeID,
		backends:   backends,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *Client) SendHeartbeat() error {
	snapshot := system.Detect()
	payload := cluster.HeartbeatRequest{
		NodeID:       c.nodeID,
		Hostname:     snapshot.Hostname,
		Arch:         snapshot.Arch,
		OS:           snapshot.OS,
		Capabilities: snapshot.Capabilities,
		Metrics:      snapshot.Metrics,
		Backends:     c.backends,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	url, err := api.JoinBasePath(c.controlURL, api.PathAgentHeartbeat)
	if err != nil {
		return err
	}
	request, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.joinToken)
	response, err := c.httpClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("control plane returned %s", response.Status)
	}
	return nil
}
