package discovery

import (
	"context"
	"fmt"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type StaticSource struct {
	Seeds     []string
	Announcer *AnnounceClient
}

func NewStaticSource(seeds []string, self SelfFunc) *StaticSource {
	return &StaticSource{
		Seeds:     normalizeSeeds(seeds),
		Announcer: NewAnnounceClient(self),
	}
}

func (s *StaticSource) Discover(ctx context.Context) ([]membership.Member, error) {
	if s == nil || len(s.Seeds) == 0 {
		return nil, nil
	}

	members := make([]membership.Member, 0)
	var errs []string
	for _, seed := range normalizeSeeds(s.Seeds) {
		discovered, err := s.Announcer.AnnounceURL(ctx, seed)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", seed, err))
			continue
		}
		members = append(members, discovered...)
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
