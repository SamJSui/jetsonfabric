package membership

import (
	"sort"
	"sync"
	"time"
)

// Store is the in-memory membership table for a jetsonfabric-node process.
type Store struct {
	mu      sync.RWMutex
	members map[string]Member
}

func NewStore() *Store {
	return &Store{members: make(map[string]Member)}
}

func (s *Store) Upsert(member Member) {
	member = Normalize(member)
	if !member.Valid() {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if existing, ok := s.members[member.NodeID]; ok {
		member = mergeMember(existing, member)
	}
	s.members[member.NodeID] = member
}

func mergeMember(existing Member, incoming Member) Member {
	incoming = mergeIdentityFields(existing, incoming)
	incoming = mergeTimeFields(existing, incoming)
	mergeRichFields(&incoming, existing)
	return Normalize(incoming)
}

func mergeIdentityFields(existing Member, incoming Member) Member {
	if incoming.Role == NodeRoleAuto && existing.Role != NodeRoleAuto {
		incoming.Role = existing.Role
	}
	if incoming.LeaderPreference == 0 && existing.LeaderPreference != 0 {
		incoming.LeaderPreference = existing.LeaderPreference
	}
	return incoming
}

func mergeTimeFields(existing Member, incoming Member) Member {
	if incoming.StartedAt.IsZero() {
		incoming.StartedAt = existing.StartedAt
	}
	if incoming.LastSeen.IsZero() {
		incoming.LastSeen = existing.LastSeen
	}
	return incoming
}

func mergeRichFields(incoming *Member, existing Member) {
	if len(incoming.Capabilities) == 0 {
		incoming.Capabilities = existing.Capabilities
	}
	if len(incoming.Metrics) == 0 {
		incoming.Metrics = existing.Metrics
	}
	if len(incoming.Engines) == 0 {
		incoming.Engines = existing.Engines
	}
}

func (s *Store) Get(nodeID string) (Member, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	member, ok := s.members[nodeID]
	return member, ok
}

func (s *Store) List() []Member {
	s.mu.RLock()
	defer s.mu.RUnlock()

	members := make([]Member, 0, len(s.members))
	for _, member := range s.members {
		members = append(members, member)
	}
	sort.Slice(members, func(i, j int) bool {
		return members[i].NodeID < members[j].NodeID
	})
	return members
}

func (s *Store) Prune(now time.Time, staleAfter time.Duration, keepNodeID string) []Member {
	s.mu.Lock()
	defer s.mu.Unlock()

	removed := []Member{}
	for nodeID, member := range s.members {
		if nodeID == keepNodeID {
			continue
		}
		if member.IsStale(now, staleAfter) {
			removed = append(removed, member)
			delete(s.members, nodeID)
		}
	}
	return removed
}
