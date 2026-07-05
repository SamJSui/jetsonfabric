package discovery

import (
	"context"
	"fmt"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type MultiSource struct {
	Sources []Source
}

func NewMultiSource(sources ...Source) MultiSource {
	filtered := make([]Source, 0, len(sources))
	for _, source := range sources {
		if source != nil {
			filtered = append(filtered, source)
		}
	}
	return MultiSource{Sources: filtered}
}

func (m MultiSource) Discover(ctx context.Context) ([]membership.Member, error) {
	membersByID := make(map[string]membership.Member)
	var membersWithoutID []membership.Member
	var errs []string

	for _, source := range m.Sources {
		if source == nil {
			continue
		}
		members, err := source.Discover(ctx)
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		for _, member := range members {
			member = membership.Normalize(member)
			if member.NodeID == "" {
				membersWithoutID = append(membersWithoutID, member)
				continue
			}
			membersByID[member.NodeID] = member
		}
	}

	out := make([]membership.Member, 0, len(membersByID)+len(membersWithoutID))
	for _, member := range membersByID {
		out = append(out, member)
	}
	out = append(out, membersWithoutID...)

	if len(out) == 0 && len(errs) > 0 {
		return nil, fmt.Errorf("discovery failed: %s", strings.Join(errs, "; "))
	}
	return out, nil
}
