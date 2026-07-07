package discovery

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const pathClusterAnnounce = "/v1/cluster/announce"

type AnnounceResponse struct {
	Leader  *membership.Member  `json:"leader,omitempty"`
	Members []membership.Member `json:"members"`
}

type AnnounceClient struct {
	Self   SelfFunc
	Client *http.Client
}

func NewAnnounceClient(self SelfFunc) *AnnounceClient {
	return &AnnounceClient{
		Self:   self,
		Client: &http.Client{Timeout: 5 * time.Second},
	}
}

func (c *AnnounceClient) Announce(ctx context.Context, peer membership.Member) ([]membership.Member, error) {
	peer = membership.Normalize(peer)
	if !peer.Valid() {
		return nil, fmt.Errorf("peer member is invalid")
	}
	return c.AnnounceURL(ctx, peer.APIURL)
}

func (c *AnnounceClient) AnnounceURL(ctx context.Context, baseURL string) ([]membership.Member, error) {
	baseURL = normalizeBaseURL(baseURL)
	if c == nil || c.Self == nil || baseURL == "" {
		return nil, nil
	}

	self := membership.Normalize(c.Self())
	if !self.Valid() {
		return nil, fmt.Errorf("self member is invalid")
	}
	if sameBaseURL(baseURL, self.APIURL) {
		return nil, nil
	}

	return c.postAnnounce(ctx, baseURL, self)
}

func (c *AnnounceClient) postAnnounce(ctx context.Context, baseURL string, self membership.Member) ([]membership.Member, error) {
	payload, err := json.Marshal(self)
	if err != nil {
		return nil, fmt.Errorf("encode announce payload: %w", err)
	}

	resp, err := c.send(ctx, baseURL+pathClusterAnnounce, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("announce to %s failed: %s", baseURL, resp.Status)
	}
	return decodeAnnounceResponse(resp)
}

func (c *AnnounceClient) send(ctx context.Context, target string, payload []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.client().Do(req)
}

func (c *AnnounceClient) client() *http.Client {
	if c.Client != nil {
		return c.Client
	}
	return &http.Client{Timeout: 5 * time.Second}
}

func decodeAnnounceResponse(resp *http.Response) ([]membership.Member, error) {
	var decoded AnnounceResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode announce response: %w", err)
	}
	return announceMembers(decoded), nil
}

func announceMembers(decoded AnnounceResponse) []membership.Member {
	members := make([]membership.Member, 0, len(decoded.Members)+1)
	members = append(members, decoded.Members...)
	if decoded.Leader != nil {
		members = append(members, *decoded.Leader)
	}
	return members
}

func normalizeBaseURL(value string) string {
	return strings.TrimRight(strings.TrimSpace(value), "/")
}

func sameBaseURL(left string, right string) bool {
	return normalizeBaseURL(left) == normalizeBaseURL(right)
}
