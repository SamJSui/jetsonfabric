package node

import (
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
)

func (cfg Config) WithBoundAPIURL(listener net.Listener) Config {
	if strings.TrimSpace(cfg.APIURL) != "" {
		return cfg
	}
	cfg.APIURL = defaultAdvertiseURLFromAddr(listener.Addr())
	return cfg
}

func (cfg Config) AdvertisePort() int {
	parsed, err := url.Parse(cfg.APIURL)
	if err == nil && parsed.Port() != "" {
		port, _ := strconv.Atoi(parsed.Port())
		if port > 0 {
			return port
		}
	}

	_, portText, err := net.SplitHostPort(cfg.Listen)
	if err != nil {
		return 0
	}

	port, _ := strconv.Atoi(portText)
	return port
}

func defaultAdvertiseURL(listen string) string {
	_, port, err := net.SplitHostPort(listen)
	if err != nil || port == "" || port == "0" {
		return ""
	}

	return "http://" + net.JoinHostPort(defaultAdvertiseHost(), port)
}

func defaultAdvertiseURLFromAddr(addr net.Addr) string {
	_, port, err := net.SplitHostPort(addr.String())
	if err != nil || port == "" || port == "0" {
		return ""
	}

	return "http://" + net.JoinHostPort(defaultAdvertiseHost(), port)
}

func defaultAdvertiseHost() string {
	host, _ := os.Hostname()
	host = strings.TrimSpace(strings.TrimSuffix(host, "."))
	if host == "" {
		return "127.0.0.1"
	}
	if host == "localhost" || strings.Contains(host, ".") {
		return host
	}
	return host + ".local"
}
