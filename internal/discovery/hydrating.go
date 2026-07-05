package discovery

import (
	"context"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type HydratingSource struct {
	Source    Source
	Announcer *AnnounceClient
}

func NewHydratingSource(source Source, announcer *AnnounceClient) *HydratingSource {
	return &HydratingSource{
		Source:    source,
		Announcer: announcer,
	}
}

func (s *HydratingSource) Discover(ctx context.Context) ([]membership.Member, error) {
	if s == nil || s.Source == nil {
		return nil, nil
	}

	peers, err := s.Source.Discover(ctx)
	if err != nil {
		return nil, err
	}
	if s.Announcer == nil {
		return peers, nil
	}

	return s.hydratePeers(ctx, peers), nil
}

func (s *HydratingSource) hydratePeers(ctx context.Context, peers []membership.Member) []membership.Member {
	membersByID := map[string]membership.Member{}
	for _, peer := range peers {
		peer = membership.Normalize(peer)
		if !peer.Valid() {
			continue
		}
		s.mergeHydratedPeer(ctx, membersByID, peer)
	}
	return membersFromMap(membersByID)
}

func (s *HydratingSource) mergeHydratedPeer(ctx context.Context, membersByID map[string]membership.Member, peer membership.Member) {
	hydrated, err := s.Announcer.Announce(ctx, peer)
	if err != nil || len(hydrated) == 0 {
		membersByID[peer.NodeID] = peer
		return
	}
	for _, member := range hydrated {
		member = membership.Normalize(member)
		if member.Valid() {
			membersByID[member.NodeID] = member
		}
	}
}

func membersFromMap(membersByID map[string]membership.Member) []membership.Member {
	members := make([]membership.Member, 0, len(membersByID))
	for _, member := range membersByID {
		members = append(members, member)
	}
	return members
}
