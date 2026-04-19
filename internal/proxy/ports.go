package proxy

import (
	"fmt"
	"net"
	"sync"
)

// PortAllocator manages port assignment for backend servers
type PortAllocator struct {
	mu      sync.Mutex
	minPort int
	maxPort int
	inUse   map[int]bool
}

// NewPortAllocator creates a new port allocator for the given range
func NewPortAllocator(minPort, maxPort int) *PortAllocator {
	return &PortAllocator{
		minPort: minPort,
		maxPort: maxPort,
		inUse:   make(map[int]bool),
	}
}

// Allocate finds and reserves an available port
func (p *PortAllocator) Allocate() (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for port := p.minPort; port <= p.maxPort; port++ {
		if p.inUse[port] {
			continue
		}

		// Check if port is actually available on the system
		if !isPortAvailable(port) {
			continue
		}

		p.inUse[port] = true
		return port, nil
	}

	return 0, fmt.Errorf("no available ports in range %d-%d", p.minPort, p.maxPort)
}

// Release frees a port for reuse
func (p *PortAllocator) Release(port int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.inUse, port)
}

// isPortAvailable checks if a port is available for binding
func isPortAvailable(port int) bool {
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	ln.Close()
	return true
}
