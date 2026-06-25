package api

import (
	"fmt"
	"net/url"
	"strings"
)

func JoinBasePath(baseURL string, path string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", fmt.Errorf("parse base url: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("base url must include scheme and host")
	}
	path = strings.Trim(path, "/")
	if path == "" {
		return parsed.String(), nil
	}
	segments := strings.Split(path, "/")
	return parsed.JoinPath(segments...).String(), nil
}
