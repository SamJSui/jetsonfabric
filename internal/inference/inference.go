// Package inference defines engine-neutral semantics for a distributed
// inference request. It owns lifecycle states, stage position arithmetic, and
// semantic payload transitions. It does not perform model execution or define
// a transport.
package inference

import "fmt"

// Phase identifies the kind of model operation performed for one pass through
// the ordered stage plan.
type Phase string

const (
	PhasePrefill Phase = "prefill"
	PhaseDecode  Phase = "decode"
)

func (p Phase) Valid() bool {
	switch p {
	case PhasePrefill, PhaseDecode:
		return true
	default:
		return false
	}
}

// SessionState describes the coordinator-visible lifecycle of one inference
// session.
type SessionState string

const (
	StateCreated    SessionState = "created"
	StatePrefilling SessionState = "prefilling"
	StateDecoding   SessionState = "decoding"
	StateFinishing  SessionState = "finishing"
	StateCompleted  SessionState = "completed"
	StateCancelled  SessionState = "cancelled"
	StateFailed     SessionState = "failed"
	StateExpired    SessionState = "expired"
)

func (s SessionState) Valid() bool {
	switch s {
	case StateCreated, StatePrefilling, StateDecoding, StateFinishing,
		StateCompleted, StateCancelled, StateFailed, StateExpired:
		return true
	default:
		return false
	}
}

func (s SessionState) Terminal() bool {
	switch s {
	case StateCompleted, StateCancelled, StateFailed, StateExpired:
		return true
	default:
		return false
	}
}

// Event requests one lifecycle transition.
type Event string

const (
	EventStartPrefill  Event = "start_prefill"
	EventBeginDecode   Event = "begin_decode"
	EventAdvanceDecode Event = "advance_decode"
	EventBeginFinish   Event = "begin_finish"
	EventComplete      Event = "complete"
	EventCancel        Event = "cancel"
	EventFail          Event = "fail"
	EventExpire        Event = "expire"
)

func (e Event) Valid() bool {
	switch e {
	case EventStartPrefill, EventBeginDecode, EventAdvanceDecode,
		EventBeginFinish, EventComplete, EventCancel, EventFail, EventExpire:
		return true
	default:
		return false
	}
}

// Transition applies an event to the current inference-session state.
func Transition(current SessionState, event Event) (SessionState, error) {
	if !current.Valid() {
		return "", fmt.Errorf("invalid inference state %q", current)
	}
	if !event.Valid() {
		return "", fmt.Errorf("invalid inference event %q", event)
	}

	var next SessionState
	switch current {
	case StateCreated:
		next = transitionCreated(event)
	case StatePrefilling:
		next = transitionPrefilling(event)
	case StateDecoding:
		next = transitionDecoding(event)
	case StateFinishing:
		next = transitionFinishing(event)
	default:
		// Terminal states do not accept additional lifecycle events.
	}
	if next == "" {
		return "", fmt.Errorf("invalid inference transition: %s --%s--> ?", current, event)
	}
	return next, nil
}

func transitionCreated(event Event) SessionState {
	switch event {
	case EventStartPrefill:
		return StatePrefilling
	case EventCancel:
		return StateCancelled
	case EventFail:
		return StateFailed
	case EventExpire:
		return StateExpired
	default:
		return ""
	}
}

func transitionPrefilling(event Event) SessionState {
	switch event {
	case EventBeginDecode:
		return StateDecoding
	case EventBeginFinish:
		return StateFinishing
	case EventCancel:
		return StateCancelled
	case EventFail:
		return StateFailed
	case EventExpire:
		return StateExpired
	default:
		return ""
	}
}

func transitionDecoding(event Event) SessionState {
	switch event {
	case EventAdvanceDecode:
		return StateDecoding
	case EventBeginFinish:
		return StateFinishing
	case EventCancel:
		return StateCancelled
	case EventFail:
		return StateFailed
	case EventExpire:
		return StateExpired
	default:
		return ""
	}
}

func transitionFinishing(event Event) SessionState {
	switch event {
	case EventComplete:
		return StateCompleted
	case EventCancel:
		return StateCancelled
	case EventFail:
		return StateFailed
	case EventExpire:
		return StateExpired
	default:
		return ""
	}
}

// StagePosition is the count-based position of a runtime in an ordered stage
// plan. There is intentionally no first/middle/last role enum.
type StagePosition struct {
	Index int
	Count int
}

func (p StagePosition) Validate() error {
	if p.Count <= 0 {
		return fmt.Errorf("stage count must be greater than zero")
	}
	if p.Index < 0 || p.Index >= p.Count {
		return fmt.Errorf("stage index %d must be in [0,%d)", p.Index, p.Count)
	}
	return nil
}

func (p StagePosition) IsFirst() bool {
	return p.Count > 0 && p.Index == 0
}

func (p StagePosition) IsLast() bool {
	return p.Count > 0 && p.Index >= 0 && p.Index < p.Count && p.Index == p.Count-1
}

func (p StagePosition) IsIntermediate() bool {
	return p.Count > 0 && p.Index > 0 && p.Index < p.Count-1
}

// PayloadKind identifies the semantic value carried between inference stages.
// Logits and KV cache remain local to the inference engine.
type PayloadKind string

const (
	PayloadKindText         PayloadKind = "text"
	PayloadKindTokens       PayloadKind = "tokens"
	PayloadKindActivation   PayloadKind = "activation"
	PayloadKindSampledToken PayloadKind = "sampled_token"
)

func (k PayloadKind) Valid() bool {
	switch k {
	case PayloadKindText, PayloadKindTokens, PayloadKindActivation, PayloadKindSampledToken:
		return true
	default:
		return false
	}
}

// AllowedInputKinds returns the legal semantic inputs for one stage operation.
// The normal external prefill path starts with text; tokens are also permitted
// for a pre-tokenized first-stage caller.
func AllowedInputKinds(phase Phase, position StagePosition) ([]PayloadKind, error) {
	if !phase.Valid() {
		return nil, fmt.Errorf("invalid inference phase %q", phase)
	}
	if err := position.Validate(); err != nil {
		return nil, err
	}
	if !position.IsFirst() {
		return []PayloadKind{PayloadKindActivation}, nil
	}
	if phase == PhasePrefill {
		return []PayloadKind{PayloadKindText, PayloadKindTokens}, nil
	}
	return []PayloadKind{PayloadKindSampledToken}, nil
}

// ExpectedOutputKind returns the required semantic output for one stage
// operation. Non-final stages emit activations; the final stage samples locally.
func ExpectedOutputKind(phase Phase, position StagePosition) (PayloadKind, error) {
	if !phase.Valid() {
		return "", fmt.Errorf("invalid inference phase %q", phase)
	}
	if err := position.Validate(); err != nil {
		return "", err
	}
	if position.IsLast() {
		return PayloadKindSampledToken, nil
	}
	return PayloadKindActivation, nil
}

// ValidatePayloadTransition checks both the input and output payload semantics
// for a stage at a specific phase and position.
func ValidatePayloadTransition(phase Phase, position StagePosition, input PayloadKind, output PayloadKind) error {
	allowed, err := AllowedInputKinds(phase, position)
	if err != nil {
		return err
	}
	if !containsPayloadKind(allowed, input) {
		return fmt.Errorf("payload %q is not a valid %s input for stage %d/%d", input, phase, position.Index, position.Count)
	}
	expectedOutput, err := ExpectedOutputKind(phase, position)
	if err != nil {
		return err
	}
	if output != expectedOutput {
		return fmt.Errorf("payload %q is not the expected %s output for stage %d/%d; want %q", output, phase, position.Index, position.Count, expectedOutput)
	}
	return nil
}

func containsPayloadKind(values []PayloadKind, expected PayloadKind) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}
