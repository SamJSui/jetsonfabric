package coordinator

import (
	"context"
	"errors"
	"sync"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

var (
	errDeploymentTransitioning = errors.New("deployment transition is in progress")
	errModelNotActive          = errors.New("requested model is not active")
	errDeploymentUnmanaged     = errors.New("no coordinator-managed deployment is active")
)

type deploymentPhase string

const (
	deploymentPhaseUnmanaged     deploymentPhase = "unmanaged"
	deploymentPhaseTransitioning deploymentPhase = "transitioning"
	deploymentPhaseActive        deploymentPhase = "active"
	deploymentPhaseFailed        deploymentPhase = "failed"
)

type deploymentSnapshot struct {
	Phase         deploymentPhase
	Active        *clusterplan.DeploymentPlan
	InFlight      int
	LastError     string
	ProposedEpoch uint64
}

type deploymentAdmission struct {
	Plan    *clusterplan.DeploymentPlan
	release func()
}

func (a deploymentAdmission) Release() {
	if a.release != nil {
		a.release()
	}
}

type deploymentState struct {
	mu        sync.Mutex
	changed   chan struct{}
	phase     deploymentPhase
	active    *clusterplan.DeploymentPlan
	inFlight  int
	lastError string
	lastEpoch uint64
}

func newDeploymentState() *deploymentState {
	return &deploymentState{
		changed: make(chan struct{}),
		phase:   deploymentPhaseUnmanaged,
	}
}

func (s *deploymentState) snapshot() deploymentSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.snapshotLocked()
}

func (s *deploymentState) snapshotLocked() deploymentSnapshot {
	var active *clusterplan.DeploymentPlan
	if s.active != nil {
		copy := *s.active
		active = &copy
	}
	return deploymentSnapshot{
		Phase:         s.phase,
		Active:        active,
		InFlight:      s.inFlight,
		LastError:     s.lastError,
		ProposedEpoch: s.lastEpoch + 1,
	}
}

func (s *deploymentState) admit(modelID string) (deploymentAdmission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.phase == deploymentPhaseTransitioning {
		return deploymentAdmission{}, errDeploymentTransitioning
	}
	if s.active != nil && s.active.Model().ModelID != modelID {
		return deploymentAdmission{}, errModelNotActive
	}
	s.inFlight++
	var plan *clusterplan.DeploymentPlan
	if s.active != nil {
		copy := *s.active
		plan = &copy
	}
	return deploymentAdmission{
		Plan: plan,
		release: func() {
			s.mu.Lock()
			if s.inFlight > 0 {
				s.inFlight--
				s.signalLocked()
			}
			s.mu.Unlock()
		},
	}, nil
}

func (s *deploymentState) beginTransition(ctx context.Context, expectedEpoch uint64) (*clusterplan.DeploymentPlan, error) {
	s.mu.Lock()
	if s.phase == deploymentPhaseTransitioning {
		s.mu.Unlock()
		return nil, errDeploymentTransitioning
	}
	if expectedEpoch != s.lastEpoch+1 {
		s.mu.Unlock()
		return nil, errors.New("deployment epoch changed while the plan was being built")
	}
	s.phase = deploymentPhaseTransitioning
	s.lastError = ""
	var previous *clusterplan.DeploymentPlan
	if s.active != nil {
		copy := *s.active
		previous = &copy
	}
	s.signalLocked()
	for s.inFlight > 0 {
		changed := s.changed
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			s.mu.Lock()
			s.phase = deploymentPhaseFailed
			s.active = nil
			s.lastError = ctx.Err().Error()
			s.signalLocked()
			s.mu.Unlock()
			return nil, ctx.Err()
		case <-changed:
			s.mu.Lock()
		}
	}
	s.mu.Unlock()
	return previous, nil
}

func (s *deploymentState) publish(plan clusterplan.DeploymentPlan) {
	s.mu.Lock()
	copy := plan
	s.active = &copy
	s.lastEpoch = plan.Identity().Epoch
	s.phase = deploymentPhaseActive
	s.lastError = ""
	s.signalLocked()
	s.mu.Unlock()
}

func (s *deploymentState) fail(err error) {
	s.mu.Lock()
	s.active = nil
	s.phase = deploymentPhaseFailed
	if err != nil {
		s.lastError = err.Error()
	}
	s.signalLocked()
	s.mu.Unlock()
}

func (s *deploymentState) signalLocked() {
	close(s.changed)
	s.changed = make(chan struct{})
}
