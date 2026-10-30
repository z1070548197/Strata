#pragma once
// STRATA_GPU_BUSY (opt-in, diagnostic): keep the compute clock up across the decode window's waits.
//
// Port of upstream's "flag-released spin kernel" (their Metal path, commit 8d3c9761).  Their rationale:
// the device is not idle between two bursts -> the next kernels run 2-3x slower.  The mechanism: while the
// host lane feeds expert data, one block of threads spins on a register-only recurrence with no memory
// traffic; the host releases it with a single write to a mapped flag, and it leaves within microseconds.
//
// THIS TOUCHES NO ARITHMETIC.  It runs on its own lowest-priority non-blocking stream, reads no model data
// and writes no model data, so a run with it on must produce the same tokens as one with it off; if the
// tokens differ, something else changed.
//
// It is not free: the block owns one SM (~1/68 of the device) and keeps the SM clock high, drawing power
// the real compute would otherwise have.  Everything about whether it pays on this machine is measured,
// not assumed - see RESULTS_2026-10-29.md SS12.57(G).
#include <string>

namespace strata {
namespace core {

// Reads STRATA_GPU_BUSY once (0 / unset: off).
bool gpu_busy_enabled();

// Per-device state for the current device.  Idempotent; returns false (with err) only when the flag is on
// and the device setup failed, which the caller should treat as "off" rather than as a run failure.
bool gpu_busy_init(std::string& err);

// Launch / release the current device's spin kernel.  Both are no-ops when the feature is off, and both
// remember cudaGetLastError() so a failure here can never surface later as a bogus error from real work.
void gpu_busy_begin();
void gpu_busy_end();

void gpu_busy_shutdown();

// RAII over one window: begin in the constructor, end in the destructor, so every early return in a
// verifier window releases the kernel.
struct GpuBusyScope {
    GpuBusyScope();
    ~GpuBusyScope();
    GpuBusyScope(const GpuBusyScope&) = delete;
    GpuBusyScope& operator=(const GpuBusyScope&) = delete;
};

}  // namespace core
}  // namespace strata
