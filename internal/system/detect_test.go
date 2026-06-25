package system

import (
	"strings"
	"testing"
)

func TestParseMemTotalGB(t *testing.T) {
	meminfo := strings.Join([]string{
		"MemFree:          123456 kB",
		"MemTotal:        8123456 kB",
		"Buffers:          123456 kB",
	}, "\n")

	got := parseMemTotalGB(strings.NewReader(meminfo))
	if got != 7.75 {
		t.Fatalf("expected 7.75 GiB, got %.2f", got)
	}
}

func TestParseMemTotalGBReturnsZeroWhenMissing(t *testing.T) {
	got := parseMemTotalGB(strings.NewReader("MemFree: 123456 kB\n"))
	if got != 0 {
		t.Fatalf("expected 0 when MemTotal is missing, got %.2f", got)
	}
}

func TestParseMemTotalGBReturnsZeroWhenMalformed(t *testing.T) {
	got := parseMemTotalGB(strings.NewReader("MemTotal: not-a-number kB\n"))
	if got != 0 {
		t.Fatalf("expected 0 for malformed MemTotal, got %.2f", got)
	}
}
