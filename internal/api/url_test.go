package api

import "testing"

func TestJoinBasePath(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		path    string
		want    string
	}{
		{
			name:    "plain base",
			baseURL: "http://127.0.0.1:8080",
			path:    PathChatCompletions,
			want:    "http://127.0.0.1:8080/v1/chat/completions",
		},
		{
			name:    "trailing slash",
			baseURL: "http://127.0.0.1:8080/",
			path:    PathChatCompletions,
			want:    "http://127.0.0.1:8080/v1/chat/completions",
		},
		{
			name:    "base path",
			baseURL: "http://127.0.0.1:8080/proxy",
			path:    PathChatCompletions,
			want:    "http://127.0.0.1:8080/proxy/v1/chat/completions",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := JoinBasePath(test.baseURL, test.path)
			if err != nil {
				t.Fatalf("JoinBasePath returned error: %v", err)
			}
			if got != test.want {
				t.Fatalf("expected %q, got %q", test.want, got)
			}
		})
	}
}

func TestJoinBasePathRejectsMissingHost(t *testing.T) {
	_, err := JoinBasePath("127.0.0.1:8080", PathChatCompletions)
	if err == nil {
		t.Fatal("expected missing scheme/host error")
	}
}
