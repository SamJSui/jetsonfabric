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
		if member.StartedAt.IsZero() {
			member.StartedAt = existing.StartedAt
		}
		if member.LastSeen.IsZero() {
			member.LastSeen = existing.LastSeen
		}
	}
	s.members[member.NodeID] = member
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
