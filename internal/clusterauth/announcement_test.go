package clusterauth

import (
	"testing"
	"time"
)

func TestAnnouncementSignatureBindsTimeTokenAndPayload(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	payload := []byte(`{"node_id":"dopey"}`)
	nonce := "request-nonce"
	timestamp, signature := SignAnnouncementRequest("cluster-secret", now, nonce, payload)
	if !VerifyAnnouncementRequest("cluster-secret", timestamp, nonce, signature, payload, now) {
		t.Fatal("valid announcement signature was rejected")
	}
	if VerifyAnnouncementRequest("wrong-secret", timestamp, nonce, signature, payload, now) {
		t.Fatal("announcement signature accepted the wrong token")
	}
	if VerifyAnnouncementRequest("cluster-secret", timestamp, nonce, signature, []byte(`{"node_id":"grumpy"}`), now) {
		t.Fatal("announcement signature accepted a modified payload")
	}
	if VerifyAnnouncementRequest("cluster-secret", timestamp, "different-nonce", signature, payload, now) {
		t.Fatal("announcement signature accepted a different nonce")
	}
	if VerifyAnnouncementRequest("cluster-secret", timestamp, nonce, signature, payload, now.Add(time.Minute)) {
		t.Fatal("expired announcement signature was accepted")
	}
}

func TestAnnouncementResponseSignatureUsesSeparateDomain(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	payload := []byte(`{"members":[]}`)
	timestamp, signature := SignAnnouncementResponse("cluster-secret", now, "nonce", payload)
	if !VerifyAnnouncementResponse("cluster-secret", timestamp, "nonce", signature, payload, now) {
		t.Fatal("valid announcement response signature was rejected")
	}
	if VerifyAnnouncementRequest("cluster-secret", timestamp, "nonce", signature, payload, now) {
		t.Fatal("response signature was accepted as a request signature")
	}
}
