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

type SelfFunc func() membership.Member

type StaticSource struct {
	Seeds  []string
	Self   SelfFunc
	Client *http.Client
}

type AnnounceResponse struct {
	Leader  *membership.Member  `json:"leader,omitempty"`
	Members []membership.Member `json:"members"`
}

func NewStaticSource(seeds []string, self SelfFunc) *StaticSource {
	return &StaticSource{
		Seeds:  normalizeSeeds(seeds),
		Self:   self,
		Client: &http.Client{Timeout: 5 * time.Second},
	}
}

func (s *StaticSource) Discover(ctx context.Context) ([]membership.Member, error) {
	if s == nil || s.Self == nil || len(s.Seeds) == 0 {
		return nil, nil
	}
	client := s.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}

	self := s.Self()
	payload, err := json.Marshal(self)
	if err != nil {
		return nil, fmt.Errorf("encode announce payload: %w", err)
	}

	members := make([]membership.Member, 0)
	var errs []string
	for _, seed := range normalizeSeeds(s.Seeds) {
		if strings.TrimRight(seed, "/") == strings.TrimRight(self.APIURL, "/") {
			continue
		}
		url := strings.TrimRight(seed, "/") + pathClusterAnnounce
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", seed, err))
			continue
		}
		func() {
			defer resp.Body.Close()
			if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
				errs = append(errs, fmt.Sprintf("%s: %s", seed, resp.Status))
				return
			}
			var decoded AnnounceResponse
			if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
				errs = append(errs, fmt.Sprintf("%s: decode response: %v", seed, err))
				return
			}
			members = append(members, decoded.Members...)
			if decoded.Leader != nil {
				members = append(members, *decoded.Leader)
			}
		}()
	}

	if len(members) == 0 && len(errs) > 0 {
		return nil, fmt.Errorf("static discovery failed: %s", strings.Join(errs, "; "))
	}
	return members, nil
}

func normalizeSeeds(seeds []string) []string {
	out := make([]string, 0, len(seeds))
	seen := map[string]bool{}
	for _, seed := range seeds {
		seed = strings.TrimSpace(seed)
		if seed == "" || seen[seed] {
			continue
		}
		seen[seed] = true
		out = append(out, seed)
	}
	return out
}
