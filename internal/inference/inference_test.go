package inference

import "testing"

func TestSessionLifecycleWithDecodeLoop(t *testing.T) {
	state := StateCreated
	state = mustTransition(t, state, EventStartPrefill, StatePrefilling)
	state = mustTransition(t, state, EventBeginDecode, StateDecoding)
	state = mustTransition(t, state, EventAdvanceDecode, StateDecoding)
	state = mustTransition(t, state, EventBeginFinish, StateFinishing)
	state = mustTransition(t, state, EventComplete, StateCompleted)
	if !state.Terminal() {
		t.Fatalf("completed state should be terminal")
	}
}

func TestSessionLifecycleCanFinishAfterPrefill(t *testing.T) {
	state := mustTransition(t, StateCreated, EventStartPrefill, StatePrefilling)
	state = mustTransition(t, state, EventBeginFinish, StateFinishing)
	mustTransition(t, state, EventComplete, StateCompleted)
}

func TestActiveStatesSupportFailureCancellationAndExpiration(t *testing.T) {
	tests := []struct {
		state SessionState
		event Event
		want  SessionState
	}{
		{StateCreated, EventCancel, StateCancelled},
		{StateCreated, EventFail, StateFailed},
		{StateCreated, EventExpire, StateExpired},
		{StatePrefilling, EventCancel, StateCancelled},
		{StatePrefilling, EventFail, StateFailed},
		{StatePrefilling, EventExpire, StateExpired},
		{StateDecoding, EventCancel, StateCancelled},
		{StateDecoding, EventFail, StateFailed},
		{StateDecoding, EventExpire, StateExpired},
		{StateFinishing, EventCancel, StateCancelled},
		{StateFinishing, EventFail, StateFailed},
		{StateFinishing, EventExpire, StateExpired},
	}
	for _, test := range tests {
		if got, err := Transition(test.state, test.event); err != nil || got != test.want {
			t.Fatalf("Transition(%q, %q) = %q, %v; want %q", test.state, test.event, got, err, test.want)
		}
	}
}

func TestInvalidAndTerminalTransitionsAreRejected(t *testing.T) {
	tests := []struct {
		state SessionState
		event Event
	}{
		{StateCreated, EventAdvanceDecode},
		{StatePrefilling, EventComplete},
		{StateDecoding, EventStartPrefill},
		{StateCompleted, EventStartPrefill},
		{StateCancelled, EventBeginDecode},
		{StateFailed, EventComplete},
		{StateExpired, EventStartPrefill},
		{SessionState("unknown"), EventStartPrefill},
		{StateCreated, Event("unknown")},
	}
	for _, test := range tests {
		if _, err := Transition(test.state, test.event); err == nil {
			t.Fatalf("Transition(%q, %q) succeeded, want error", test.state, test.event)
		}
	}
}

func TestStagePositionArithmetic(t *testing.T) {
	tests := []struct {
		position     StagePosition
		first        bool
		last         bool
		intermediate bool
	}{
		{StagePosition{Index: 0, Count: 1}, true, true, false},
		{StagePosition{Index: 0, Count: 3}, true, false, false},
		{StagePosition{Index: 1, Count: 3}, false, false, true},
		{StagePosition{Index: 2, Count: 3}, false, true, false},
	}
	for _, test := range tests {
		if err := test.position.Validate(); err != nil {
			t.Fatalf("valid position %+v rejected: %v", test.position, err)
		}
		if test.position.IsFirst() != test.first || test.position.IsLast() != test.last || test.position.IsIntermediate() != test.intermediate {
			t.Fatalf("unexpected position classification for %+v", test.position)
		}
	}

	for _, position := range []StagePosition{{Index: 0, Count: 0}, {Index: -1, Count: 2}, {Index: 2, Count: 2}} {
		if err := position.Validate(); err == nil {
			t.Fatalf("invalid position %+v accepted", position)
		}
	}
}

func TestPayloadContractsAcrossStageCounts(t *testing.T) {
	for stageCount := 1; stageCount <= 8; stageCount++ {
		for stageIndex := 0; stageIndex < stageCount; stageIndex++ {
			position := StagePosition{Index: stageIndex, Count: stageCount}

			prefillInput := PayloadKindActivation
			decodeInput := PayloadKindActivation
			if position.IsFirst() {
				prefillInput = PayloadKindText
				decodeInput = PayloadKindSampledToken
			}
			output := PayloadKindActivation
			if position.IsLast() {
				output = PayloadKindSampledToken
			}

			if err := ValidatePayloadTransition(PhasePrefill, position, prefillInput, output); err != nil {
				t.Fatalf("prefill stage %d/%d rejected: %v", stageIndex, stageCount, err)
			}
			if err := ValidatePayloadTransition(PhaseDecode, position, decodeInput, output); err != nil {
				t.Fatalf("decode stage %d/%d rejected: %v", stageIndex, stageCount, err)
			}
		}
	}
}

func TestPrefillFirstStageAllowsPretokenizedInput(t *testing.T) {
	position := StagePosition{Index: 0, Count: 2}
	if err := ValidatePayloadTransition(PhasePrefill, position, PayloadKindTokens, PayloadKindActivation); err != nil {
		t.Fatalf("pretokenized prefill rejected: %v", err)
	}
}

func TestPayloadContractRejectsWrongKinds(t *testing.T) {
	tests := []struct {
		phase    Phase
		position StagePosition
		input    PayloadKind
		output   PayloadKind
	}{
		{PhasePrefill, StagePosition{Index: 0, Count: 2}, PayloadKindActivation, PayloadKindActivation},
		{PhaseDecode, StagePosition{Index: 0, Count: 2}, PayloadKindText, PayloadKindActivation},
		{PhasePrefill, StagePosition{Index: 1, Count: 2}, PayloadKindActivation, PayloadKindActivation},
		{PhaseDecode, StagePosition{Index: 1, Count: 2}, PayloadKindActivation, PayloadKindActivation},
		{Phase("unknown"), StagePosition{Index: 0, Count: 1}, PayloadKindText, PayloadKindSampledToken},
		{PhasePrefill, StagePosition{Index: 1, Count: 1}, PayloadKindText, PayloadKindSampledToken},
	}
	for _, test := range tests {
		if err := ValidatePayloadTransition(test.phase, test.position, test.input, test.output); err == nil {
			t.Fatalf("invalid transition accepted: %+v", test)
		}
	}
}

func mustTransition(t *testing.T, current SessionState, event Event, want SessionState) SessionState {
	t.Helper()
	got, err := Transition(current, event)
	if err != nil {
		t.Fatalf("Transition(%q, %q): %v", current, event, err)
	}
	if got != want {
		t.Fatalf("Transition(%q, %q) = %q, want %q", current, event, got, want)
	}
	return got
}
