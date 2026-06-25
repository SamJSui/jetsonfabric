package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/SamJSui/JetsonMesh/internal/cluster"
	"github.com/SamJSui/JetsonMesh/internal/system"
)

type Client struct {
	controlURL string
	joinToken  string
	nodeID     string
	httpClient *http.Client
}

func NewClient(controlURL string, joinToken string, nodeID string) *Client {
	return &Client{
		controlURL: strings.TrimRight(controlURL, "/"),
		joinToken:  joinToken,
		nodeID:     nodeID,
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
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequest(http.MethodPost, c.controlURL+"/v1/agents/heartbeat", bytes.NewReader(body))
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
