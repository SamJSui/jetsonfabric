package discovery

import (
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	mdnsPort          = 5353
	mdnsMaxPacketSize = 9000
	mdnsTTLSeconds    = 30

	dnsTypePTR = 12
	dnsTypeTXT = 16
	dnsTypeSRV = 33
	dnsClassIN = 1
)

var mdnsIPv4 = net.IPv4(224, 0, 0, 251)

type MDNSConfig struct {
	ClusterID     string
	Service       string
	Domain        string
	Port          int
	BrowseTimeout time.Duration
	Self          SelfFunc
}

type MDNSSource struct {
	Config MDNSConfig
}

type MDNSAdvertiser struct {
	Config MDNSConfig
	conn   *net.UDPConn
}

func NewMDNSSource(cfg MDNSConfig) *MDNSSource {
	return &MDNSSource{Config: normalizeMDNSConfig(cfg)}
}

func NewMDNSAdvertiser(cfg MDNSConfig) *MDNSAdvertiser {
	return &MDNSAdvertiser{Config: normalizeMDNSConfig(cfg)}
}

func (s *MDNSSource) Discover(ctx context.Context) ([]membership.Member, error) {
	if s == nil {
		return nil, nil
	}
	cfg := normalizeMDNSConfig(s.Config)
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4zero, Port: 0})
	if err != nil {
		return nil, fmt.Errorf("open mdns browse socket: %w", err)
	}
	defer conn.Close()

	deadline := time.Now().Add(cfg.BrowseTimeout)
	if ctxDeadline, ok := ctx.Deadline(); ok && ctxDeadline.Before(deadline) {
		deadline = ctxDeadline
	}
	if err := conn.SetDeadline(deadline); err != nil {
		return nil, fmt.Errorf("set mdns browse deadline: %w", err)
	}

	query := buildDNSQuery(serviceFQDN(cfg))
	if _, err := conn.WriteToUDP(query, mdnsAddress()); err != nil {
		return nil, fmt.Errorf("send mdns query: %w", err)
	}

	membersByID := map[string]membership.Member{}
	buf := make([]byte, mdnsMaxPacketSize)
	for {
		n, remote, err := conn.ReadFromUDP(buf)
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				break
			}
			if ctx.Err() != nil {
				break
			}
			return nil, fmt.Errorf("read mdns response: %w", err)
		}
		for _, txt := range parseTXTAnswers(buf[:n]) {
			member, ok := memberFromMDNSTXT(txt, remote.IP.String())
			if !ok {
				continue
			}
			if cfg.ClusterID != "" && member.ClusterID != cfg.ClusterID {
				continue
			}
			if member.NodeID == "" {
				continue
			}
			membersByID[member.NodeID] = member
		}
	}

	members := make([]membership.Member, 0, len(membersByID))
	for _, member := range membersByID {
		members = append(members, member)
	}
	return members, nil
}

func (a *MDNSAdvertiser) Start(ctx context.Context) error {
	if a == nil || a.Config.Self == nil {
		return nil
	}
	cfg := normalizeMDNSConfig(a.Config)
	conn, err := net.ListenMulticastUDP("udp4", nil, mdnsAddress())
	if err != nil {
		return fmt.Errorf("listen mdns multicast: %w", err)
	}
	if err := conn.SetReadBuffer(mdnsMaxPacketSize); err != nil {
		_ = conn.Close()
		return fmt.Errorf("set mdns read buffer: %w", err)
	}
	a.Config = cfg
	a.conn = conn

	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()
	go a.serve(ctx, conn)
	_ = a.sendResponse(conn, mdnsAddress())
	return nil
}

func (a *MDNSAdvertiser) Shutdown() {
	if a == nil || a.conn == nil {
		return
	}
	_ = a.conn.Close()
	a.conn = nil
}

func (a *MDNSAdvertiser) serve(ctx context.Context, conn *net.UDPConn) {
	buf := make([]byte, mdnsMaxPacketSize)
	serviceName := serviceFQDN(a.Config)
	for {
		n, remote, err := conn.ReadFromUDP(buf)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			continue
		}
		if !packetAsksForService(buf[:n], serviceName) {
			continue
		}
		_ = a.sendResponse(conn, remote)
	}
}

func (a *MDNSAdvertiser) sendResponse(conn *net.UDPConn, target *net.UDPAddr) error {
	member := a.Config.Self()
	packet := buildDNSResponse(member, a.Config)
	_, err := conn.WriteToUDP(packet, target)
	return err
}

func normalizeMDNSConfig(cfg MDNSConfig) MDNSConfig {
	cfg.ClusterID = strings.TrimSpace(cfg.ClusterID)
	cfg.Service = strings.TrimSpace(cfg.Service)
	cfg.Domain = strings.TrimSpace(cfg.Domain)
	if cfg.Service == "" {
		cfg.Service = DefaultMDNSService
	}
	cfg.Service = strings.TrimSuffix(cfg.Service, ".")
	if cfg.Domain == "" {
		cfg.Domain = DefaultMDNSDomain
	}
	cfg.Domain = ensureTrailingDot(cfg.Domain)
	if cfg.BrowseTimeout <= 0 {
		cfg.BrowseTimeout = DefaultMDNSBrowseTimeout
	}
	return cfg
}

func serviceFQDN(cfg MDNSConfig) string {
	cfg = normalizeMDNSConfig(cfg)
	return ensureTrailingDot(strings.TrimSuffix(cfg.Service, ".") + "." + strings.TrimSuffix(cfg.Domain, "."))
}

func instanceFQDN(member membership.Member, cfg MDNSConfig) string {
	label := sanitizeDNSLabel(member.NodeName)
	if label == "" {
		label = "node"
	}
	if len(member.NodeID) >= 8 {
		label = label + "-" + member.NodeID[:8]
	}
	return ensureTrailingDot(label + "." + strings.TrimSuffix(serviceFQDN(cfg), "."))
}

func hostnameFQDN(member membership.Member, cfg MDNSConfig) string {
	host := sanitizeDNSLabel(member.Hostname)
	if host == "" {
		host = sanitizeDNSLabel(member.NodeName)
	}
	if host == "" {
		host = "jetsonfabric-node"
	}
	if !strings.Contains(host, ".") {
		host = host + "." + strings.TrimSuffix(normalizeMDNSConfig(cfg).Domain, ".")
	}
	return ensureTrailingDot(host)
}

func sanitizeDNSLabel(value string) string {
	value = strings.TrimSpace(strings.TrimSuffix(value, "."))
	if value == "" {
		return ""
	}
	var builder strings.Builder
	for _, r := range value {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			builder.WriteRune(unicode.ToLower(r))
		case r == '-', r == '_', r == '.':
			builder.WriteRune(r)
		case r == ' ':
			builder.WriteRune('-')
		}
	}
	return strings.Trim(builder.String(), "-.")
}

func ensureTrailingDot(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "."
	}
	if strings.HasSuffix(value, ".") {
		return value
	}
	return value + "."
}

func namesEqual(left, right string) bool {
	return strings.EqualFold(strings.TrimSuffix(left, "."), strings.TrimSuffix(right, "."))
}

func mdnsAddress() *net.UDPAddr {
	return &net.UDPAddr{IP: mdnsIPv4, Port: mdnsPort}
}

func buildDNSQuery(serviceName string) []byte {
	packet := make([]byte, 0, 512)
	packet = appendHeader(packet, 0, 0, 1, 0, 0, 0)
	packet = appendName(packet, serviceName)
	packet = appendUint16(packet, dnsTypePTR)
	packet = appendUint16(packet, dnsClassIN)
	return packet
}

func buildDNSResponse(member membership.Member, cfg MDNSConfig) []byte {
	cfg = normalizeMDNSConfig(cfg)
	serviceName := serviceFQDN(cfg)
	instanceName := instanceFQDN(member, cfg)
	hostName := hostnameFQDN(member, cfg)
	port := cfg.Port
	if port <= 0 {
		port = portFromURL(member.APIURL)
	}

	packet := make([]byte, 0, 1024)
	packet = appendHeader(packet, 0, 0x8400, 0, 3, 0, 0)
	packet = appendResourceRecord(packet, serviceName, dnsTypePTR, mdnsTTLSeconds, appendName(nil, instanceName))

	srv := make([]byte, 0, 64)
	srv = appendUint16(srv, 0)
	srv = appendUint16(srv, 0)
	srv = appendUint16(srv, port)
	srv = appendName(srv, hostName)
	packet = appendResourceRecord(packet, instanceName, dnsTypeSRV, mdnsTTLSeconds, srv)
	packet = appendResourceRecord(packet, instanceName, dnsTypeTXT, mdnsTTLSeconds, encodeTXT(memberTXT(member, port)))
	return packet
}

func appendHeader(packet []byte, id uint16, flags uint16, qdCount uint16, anCount uint16, nsCount uint16, arCount uint16) []byte {
	packet = appendUint16(packet, int(id))
	packet = appendUint16(packet, int(flags))
	packet = appendUint16(packet, int(qdCount))
	packet = appendUint16(packet, int(anCount))
	packet = appendUint16(packet, int(nsCount))
	packet = appendUint16(packet, int(arCount))
	return packet
}

func appendResourceRecord(packet []byte, name string, recordType int, ttl int, data []byte) []byte {
	packet = appendName(packet, name)
	packet = appendUint16(packet, recordType)
	packet = appendUint16(packet, dnsClassIN)
	packet = appendUint32(packet, ttl)
	packet = appendUint16(packet, len(data))
	packet = append(packet, data...)
	return packet
}

func appendName(packet []byte, name string) []byte {
	name = strings.TrimSuffix(name, ".")
	if name == "" {
		return append(packet, 0)
	}
	for _, label := range strings.Split(name, ".") {
		if label == "" {
			continue
		}
		if len(label) > 63 {
			label = label[:63]
		}
		packet = append(packet, byte(len(label)))
		packet = append(packet, label...)
	}
	return append(packet, 0)
}

func appendUint16(packet []byte, value int) []byte {
	return append(packet, byte(value>>8), byte(value))
}

func appendUint32(packet []byte, value int) []byte {
	return append(packet, byte(value>>24), byte(value>>16), byte(value>>8), byte(value))
}

func packetAsksForService(packet []byte, serviceName string) bool {
	if len(packet) < 12 {
		return false
	}
	flags := binary.BigEndian.Uint16(packet[2:4])
	if flags&0x8000 != 0 {
		return false
	}
	questionCount := int(binary.BigEndian.Uint16(packet[4:6]))
	offset := 12
	for i := 0; i < questionCount; i++ {
		name, next, ok := readName(packet, offset)
		if !ok || next+4 > len(packet) {
			return false
		}
		recordType := int(binary.BigEndian.Uint16(packet[next : next+2]))
		offset = next + 4
		if recordType == dnsTypePTR && namesEqual(name, serviceName) {
			return true
		}
	}
	return false
}

func parseTXTAnswers(packet []byte) [][]string {
	if len(packet) < 12 {
		return nil
	}
	questionCount := int(binary.BigEndian.Uint16(packet[4:6]))
	answerCount := int(binary.BigEndian.Uint16(packet[6:8])) +
		int(binary.BigEndian.Uint16(packet[8:10])) +
		int(binary.BigEndian.Uint16(packet[10:12]))
	offset := 12
	for i := 0; i < questionCount; i++ {
		_, next, ok := readName(packet, offset)
		if !ok || next+4 > len(packet) {
			return nil
		}
		offset = next + 4
	}

	var records [][]string
	for i := 0; i < answerCount; i++ {
		_, next, ok := readName(packet, offset)
		if !ok || next+10 > len(packet) {
			return records
		}
		recordType := int(binary.BigEndian.Uint16(packet[next : next+2]))
		dataLength := int(binary.BigEndian.Uint16(packet[next+8 : next+10]))
		dataStart := next + 10
		dataEnd := dataStart + dataLength
		if dataEnd > len(packet) {
			return records
		}
		if recordType == dnsTypeTXT {
			records = append(records, decodeTXT(packet[dataStart:dataEnd]))
		}
		offset = dataEnd
	}
	return records
}

func readName(packet []byte, offset int) (string, int, bool) {
	labels := make([]string, 0, 4)
	seen := 0
	next := offset
	jumped := false
	for {
		if offset >= len(packet) || seen > len(packet) {
			return "", 0, false
		}
		seen++
		length := int(packet[offset])
		if length&0xC0 == 0xC0 {
			if offset+1 >= len(packet) {
				return "", 0, false
			}
			pointer := int(binary.BigEndian.Uint16(packet[offset:offset+2]) & 0x3FFF)
			if !jumped {
				next = offset + 2
			}
			offset = pointer
			jumped = true
			continue
		}
		if length&0xC0 != 0 {
			return "", 0, false
		}
		if length == 0 {
			if !jumped {
				next = offset + 1
			}
			break
		}
		offset++
		if offset+length > len(packet) {
			return "", 0, false
		}
		labels = append(labels, string(packet[offset:offset+length]))
		offset += length
	}
	return ensureTrailingDot(strings.Join(labels, ".")), next, true
}

func encodeTXT(records []string) []byte {
	out := make([]byte, 0)
	for _, record := range records {
		if record == "" {
			continue
		}
		if len(record) > 255 {
			record = record[:255]
		}
		out = append(out, byte(len(record)))
		out = append(out, record...)
	}
	return out
}

func decodeTXT(data []byte) []string {
	records := make([]string, 0)
	for offset := 0; offset < len(data); {
		length := int(data[offset])
		offset++
		if offset+length > len(data) {
			break
		}
		records = append(records, string(data[offset:offset+length]))
		offset += length
	}
	return records
}

func memberTXT(member membership.Member, port int) []string {
	records := []string{
		"cluster_id=" + member.ClusterID,
		"node_id=" + member.NodeID,
		"node_name=" + member.NodeName,
		"hostname=" + member.Hostname,
		"api_url=" + member.APIURL,
		"runtime_url=" + member.RuntimeURL,
		"control_eligible=" + strconv.FormatBool(member.ControlEligible),
		"control_priority=" + strconv.Itoa(member.ControlPriority),
		"arch=" + member.Arch,
		"os=" + string(member.OS),
		"started_at=" + member.StartedAt.Format(time.RFC3339Nano),
	}
	if port > 0 {
		records = append(records, "api_port="+strconv.Itoa(port))
	}
	return records
}

func memberFromMDNSTXT(records []string, sourceIP string) (membership.Member, bool) {
	values := txtValues(records)
	port, _ := strconv.Atoi(values["api_port"])
	apiURL := strings.TrimSpace(values["api_url"])
	if sourceIP != "" && port > 0 {
		apiURL = fmt.Sprintf("http://%s:%d", sourceIP, port)
	}
	controlEligible, _ := strconv.ParseBool(values["control_eligible"])
	controlPriority, _ := strconv.Atoi(values["control_priority"])
	startedAt, _ := time.Parse(time.RFC3339Nano, values["started_at"])

	member := membership.Member{
		ClusterID:       values["cluster_id"],
		NodeID:          values["node_id"],
		NodeName:        values["node_name"],
		Hostname:        values["hostname"],
		APIURL:          apiURL,
		RuntimeURL:      values["runtime_url"],
		ControlEligible: controlEligible,
		ControlPriority: controlPriority,
		Arch:            values["arch"],
		OS:              cluster.OperatingSystem(values["os"]),
		StartedAt:       startedAt,
		LastSeen:        time.Now().UTC(),
	}
	member = membership.Normalize(member)
	return member, member.Valid()
}

func txtValues(records []string) map[string]string {
	values := make(map[string]string, len(records))
	for _, record := range records {
		key, value, ok := strings.Cut(record, "=")
		if !ok {
			continue
		}
		values[strings.TrimSpace(key)] = strings.TrimSpace(value)
	}
	return values
}

func portFromURL(rawURL string) int {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return 0
	}
	port := parsed.Port()
	if port == "" {
		return 0
	}
	value, _ := strconv.Atoi(port)
	return value
}
