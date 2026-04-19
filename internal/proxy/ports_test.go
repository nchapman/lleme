package proxy

import (
	"testing"
)

func TestPortAllocator(t *testing.T) {
	allocator := NewPortAllocator(59000, 59005)

	// Should be able to allocate ports
	port1, err := allocator.Allocate()
	if err != nil {
		t.Fatalf("First allocation failed: %v", err)
	}
	if port1 < 59000 || port1 > 59005 {
		t.Errorf("Port %d outside range 59000-59005", port1)
	}

	// Distinct ports on subsequent allocations
	port2, err := allocator.Allocate()
	if err != nil {
		t.Fatalf("Second allocation failed: %v", err)
	}
	if port2 == port1 {
		t.Errorf("Second allocation returned duplicate port %d", port2)
	}

	// Release a port and confirm it can be reallocated
	allocator.Release(port2)
	port3, err := allocator.Allocate()
	if err != nil {
		t.Fatalf("Allocation after release failed: %v", err)
	}
	if port3 < 59000 || port3 > 59005 {
		t.Errorf("Reallocated port %d outside range", port3)
	}
}

func TestPortAllocatorExhaustion(t *testing.T) {
	// Very small range for testing exhaustion
	allocator := NewPortAllocator(59100, 59101)

	// Allocate all available ports
	_, err1 := allocator.Allocate()
	_, err2 := allocator.Allocate()

	if err1 != nil || err2 != nil {
		t.Skip("Ports 59100-59101 not available for testing")
	}

	// Third allocation should fail (range exhausted)
	_, err := allocator.Allocate()
	if err == nil {
		t.Error("Expected error when port range exhausted")
	}
}
