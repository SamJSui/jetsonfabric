package clusterauth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"time"
)

const (
	HeaderAnnouncementNonce             = "X-JetsonFabric-Announce-Nonce"
	HeaderAnnouncementTimestamp         = "X-JetsonFabric-Announce-Timestamp"
	HeaderAnnouncementSignature         = "X-JetsonFabric-Announce-Signature"
	HeaderAnnouncementResponseTimestamp = "X-JetsonFabric-Announce-Response-Timestamp"
	HeaderAnnouncementResponseSignature = "X-JetsonFabric-Announce-Response-Signature"
	announcementMaxClockSkew            = 30 * time.Second
)

func SignAnnouncementRequest(token string, at time.Time, nonce string, payload []byte) (string, string) {
	return signAnnouncement("request", token, at, nonce, payload)
}

func VerifyAnnouncementRequest(token, timestamp, nonce, signature string, payload []byte, now time.Time) bool {
	return verifyAnnouncement("request", token, timestamp, nonce, signature, payload, now)
}

func SignAnnouncementResponse(token string, at time.Time, nonce string, payload []byte) (string, string) {
	return signAnnouncement("response", token, at, nonce, payload)
}

func VerifyAnnouncementResponse(token, timestamp, nonce, signature string, payload []byte, now time.Time) bool {
	return verifyAnnouncement("response", token, timestamp, nonce, signature, payload, now)
}

func signAnnouncement(kind, token string, at time.Time, nonce string, payload []byte) (string, string) {
	timestamp := strconv.FormatInt(at.UTC().Unix(), 10)
	mac := hmac.New(sha256.New, []byte(token))
	_, _ = mac.Write(announcementMessage(kind, timestamp, nonce, payload))
	return timestamp, hex.EncodeToString(mac.Sum(nil))
}

func verifyAnnouncement(kind, token, timestamp, nonce, signature string, payload []byte, now time.Time) bool {
	if nonce == "" {
		return false
	}
	seconds, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return false
	}
	signedAt := time.Unix(seconds, 0)
	skew := now.UTC().Sub(signedAt)
	if skew < -announcementMaxClockSkew || skew > announcementMaxClockSkew {
		return false
	}
	provided, err := hex.DecodeString(signature)
	if err != nil {
		return false
	}
	mac := hmac.New(sha256.New, []byte(token))
	_, _ = mac.Write(announcementMessage(kind, timestamp, nonce, payload))
	return hmac.Equal(provided, mac.Sum(nil))
}

func announcementMessage(kind, timestamp, nonce string, payload []byte) []byte {
	digest := sha256.Sum256(payload)
	message := make([]byte, 0, len(kind)+len(timestamp)+len(nonce)+len(digest)+64)
	message = append(message, "jetsonfabric-announce-v1\n"...)
	message = append(message, kind...)
	message = append(message, '\n')
	message = append(message, timestamp...)
	message = append(message, '\n')
	message = append(message, nonce...)
	message = append(message, '\n')
	message = append(message, digest[:]...)
	return message
}
