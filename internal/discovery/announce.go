package discovery

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterauth"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	pathClusterAnnounce         = "/v1/cluster/announce"
	maxAnnouncementResponseSize = 1 << 20
	announcementNonceBytes      = 16
)

type AnnounceResponse struct {
	Leader  *membership.Member  `json:"leader,omitempty"`
	Members []membership.Member `json:"members"`
}

type AnnounceClient struct {
	Self         SelfFunc
	ClusterToken string
	Client       *http.Client
}

func NewAnnounceClient(self SelfFunc, clusterToken string) *AnnounceClient {
	return &AnnounceClient{
		Self:         self,
		ClusterToken: strings.TrimSpace(clusterToken),
		Client:       announcementHTTPClient(),
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

	resp, nonce, err := c.send(ctx, baseURL+pathClusterAnnounce, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("announce to %s failed: %s", baseURL, resp.Status)
	}
	return decodeAnnounceResponse(resp, c.ClusterToken, nonce)
}

func (c *AnnounceClient) send(ctx context.Context, target string, payload []byte) (*http.Response, string, error) {
	nonce, err := newAnnouncementNonce()
	if err != nil {
		return nil, "", fmt.Errorf("create announce nonce: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(payload))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.ClusterToken != "" {
		timestamp, signature := clusterauth.SignAnnouncementRequest(
			c.ClusterToken,
			time.Now().UTC(),
			nonce,
			payload,
		)
		req.Header.Set(clusterauth.HeaderAnnouncementNonce, nonce)
		req.Header.Set(clusterauth.HeaderAnnouncementTimestamp, timestamp)
		req.Header.Set(clusterauth.HeaderAnnouncementSignature, signature)
	}
	client := *c.client()
	client.CheckRedirect = rejectAnnouncementRedirect
	resp, err := client.Do(req)
	return resp, nonce, err
}

func (c *AnnounceClient) client() *http.Client {
	if c.Client != nil {
		return c.Client
	}
	return announcementHTTPClient()
}

func announcementHTTPClient() *http.Client {
	return &http.Client{Timeout: 5 * time.Second, CheckRedirect: rejectAnnouncementRedirect}
}

func rejectAnnouncementRedirect(*http.Request, []*http.Request) error {
	return http.ErrUseLastResponse
}

func decodeAnnounceResponse(resp *http.Response, clusterToken, nonce string) ([]membership.Member, error) {
	payload, err := io.ReadAll(io.LimitReader(resp.Body, maxAnnouncementResponseSize+1))
	if err != nil {
		return nil, fmt.Errorf("read announce response: %w", err)
	}
	if len(payload) > maxAnnouncementResponseSize {
		return nil, fmt.Errorf("announce response exceeds 1 MiB")
	}
	if clusterToken != "" && !clusterauth.VerifyAnnouncementResponse(
		clusterToken,
		resp.Header.Get(clusterauth.HeaderAnnouncementResponseTimestamp),
		nonce,
		resp.Header.Get(clusterauth.HeaderAnnouncementResponseSignature),
		payload,
		time.Now().UTC(),
	) {
		return nil, fmt.Errorf("announce response signature is invalid")
	}
	var decoded AnnounceResponse
	if err := json.NewDecoder(bytes.NewReader(payload)).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode announce response: %w", err)
	}
	return announceMembers(decoded), nil
}

func newAnnouncementNonce() (string, error) {
	nonce := make([]byte, announcementNonceBytes)
	if _, err := rand.Read(nonce); err != nil {
		return "", err
	}
	return hex.EncodeToString(nonce), nil
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
