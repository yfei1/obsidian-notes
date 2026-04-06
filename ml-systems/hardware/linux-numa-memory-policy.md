# Linux NUMA Memory Policy
#ml-systems #hardware #interview-prep

## TL;DR

Linux provides two independent knobs for NUMA locality: **CPU affinity**
(`sched_setaffinity` — which cores a thread runs on) and **memory policy**
(`set_mempolicy` — which NUMA nodes the page allocator draws from). They don't
interact automatically — setting one does NOT set the other. The standard pattern
for ML serving: pin CPUs first (`sched_setaffinity`), then set preferred node
(`numa_set_preferred`). Hard bind (`numa_set_membind`) risks OOM kill.

---

## Core Intuition

**The problem**: on a 2-socket server, a thread can run on any CPU and allocate
memory on any NUMA node. Without explicit control, the OS scheduler may run your
GPU worker on socket 0's CPUs while its tensors live on socket 1's DRAM — every
memory access pays a ~1.8× latency penalty.

**Two independent fixes**:
1. **CPU affinity**: tell the scheduler "only run this thread on socket 0's CPUs"
2. **Memory policy**: tell the page allocator "only allocate pages from socket 0's DRAM"

These are orthogonal kernel mechanisms. Setting CPU affinity does not change memory
policy, and vice versa. But they compose: CPU affinity + default memory policy gives
you first-touch locality (pages fault in on the local node because the thread only
runs on local CPUs).

---

## How It Works

### The two control planes

| Mechanism | Syscall | Kernel subsystem | Controls |
|-----------|---------|-----------------|----------|
| CPU affinity | `sched_setaffinity(2)` | Scheduler | Which CPUs a thread is eligible to run on |
| Memory policy | `set_mempolicy(2)` | Page allocator (mempolicy) | Which NUMA nodes new pages come from |
| Per-VMA policy | `mbind(2)` | Page allocator (per address range) | Overrides process policy for specific mappings |

### Memory policy modes

**MPOL_DEFAULT** (the default — "first-touch"):
- No explicit policy. The page allocator places new pages on the NUMA node of the
  CPU that triggers the page fault.
- With CPU affinity pinned to node 0, all first-touch faults go to node 0 — achieving
  NUMA locality without explicit memory policy.
- If the local node is full, falls back to nearest nodes by distance.

**MPOL_PREFERRED** (`numa_set_preferred` / `numactl --preferred`):
- Prefer a specific NUMA node for new allocations.
- **Soft policy**: if preferred node is full, kernel falls back to other nodes in
  distance order. Allocation always succeeds — no OOM from policy alone.
- Maps to: `set_mempolicy(MPOL_PREFERRED, nodemask, maxnodes)`.

**MPOL_BIND** (`numa_set_membind` / `numactl --membind`):
- **Hard policy**: allocate ONLY from specified nodes.
- If bound nodes are exhausted: **OOM killer fires**. No fallback to other nodes.
- Use with caution for long-running processes with large memory footprints (e.g.,
  LLM serving with large KV caches).
- Maps to: `set_mempolicy(MPOL_BIND, nodemask, maxnodes)`.

**MPOL_INTERLEAVE** (`numactl --interleave`):
- Round-robin page allocation across specified nodes.
- Optimizes for **aggregate bandwidth** over single-access latency — spreads load
  across multiple memory controllers.
- Used by kernel at boot for init-time allocations (avoids overloading node 0).

**MPOL_PREFERRED_MANY** (Linux 5.15+):
- Like MPOL_PREFERRED but with multiple preferred nodes.
- Designed for heterogeneous memory (CXL tiers) where "prefer fast memory, fall
  back to slow" requires more than one preferred node.

### libnuma convenience functions

| libnuma function | Underlying syscall(s) | CPU affinity? | Memory policy? |
|-----------------|----------------------|---------------|----------------|
| `numa_run_on_node(N)` | `sched_setaffinity` with node N's cpumask | **Yes** | No |
| `numa_set_preferred(N)` | `set_mempolicy(MPOL_PREFERRED, {N})` | No | **Yes** (soft) |
| `numa_set_membind(mask)` | `set_mempolicy(MPOL_BIND, mask)` | No | **Yes** (hard) |
| `numa_bind(mask)` | `sched_setaffinity` + `set_mempolicy(MPOL_BIND)` | **Yes** | **Yes** (hard) |
| `numa_set_interleave_mask(mask)` | `set_mempolicy(MPOL_INTERLEAVE, mask)` | No | **Yes** |

`numa_run_on_node` is just `sched_setaffinity` — it does NOT set memory policy.
To get both CPU and memory locality, call `numa_run_on_node` + `numa_set_preferred`
(safe) or use `numa_bind` (hard bind — risk OOM).

### How first-touch + CPU pinning works

The most common NUMA locality pattern in practice:

```
1. sched_setaffinity(0, node_0_cpumask)    // pin thread to node 0 CPUs
2. tensor = torch.randn(1024, 1024)        // page faults on node 0 CPUs
                                           // → pages allocated on node 0 DRAM
3. result = tensor.cuda()                  // DMA from node 0 DRAM (local to GPU 0)
```

**Order matters**: pin CPU *before* allocating memory. If you allocate first (step 2
before step 1), the thread may be running on node 1 when the page faults occur —
pages land on node 1. Pinning the CPU afterward does NOT migrate existing pages.

### set_mempolicy vs mbind: process-wide vs per-range

| Scope | Syscall | Precedence |
|-------|---------|------------|
| Process-wide | `set_mempolicy(mode, nodemask)` | Lower — applies to all new allocations |
| Per address range | `mbind(addr, len, mode, nodemask, flags)` | Higher — overrides process policy for that range |

`mbind` is useful for fine-grained control: set process-wide MPOL_INTERLEAVE for
general allocations, but `mbind(MPOL_BIND, hot_buffer_addr, len, {local_node})` for
a latency-critical data structure.

By default, `mbind` only affects new allocations in the range. With `MPOL_MF_MOVE`
flag, the kernel migrates existing pages to match the new policy.

---

## Key Trade-offs & Decisions

### preferred vs membind for ML serving

| | MPOL_PREFERRED | MPOL_BIND |
|-|----------------|-----------|
| Failure mode | Falls back to remote node (higher latency) | OOM kill |
| Locality guarantee | Best-effort | Strict |
| Risk | ~1.8× latency on spilled pages | Process death |
| Best for | Long-running servers, variable memory | Short jobs, known memory budget |

**For vLLM workers**: use MPOL_PREFERRED. A serving worker's memory footprint is
unpredictable (variable batch sizes, KV cache growth). A 1.8× penalty on a few
spilled pages is better than an OOM kill during peak load.

**For training**: MPOL_BIND may be acceptable because memory footprint is
deterministic (fixed batch size, fixed model) and can be validated before launch.

### numactl flags cheat sheet

```bash
# Pin CPU + soft memory preference (recommended for serving)
numactl --cpunodebind=0 --preferred=0 python serve.py

# Pin CPU + hard memory bind (acceptable for training)
numactl --cpunodebind=0 --membind=0 python train.py

# Interleave for bandwidth-heavy init, then rebind
numactl --interleave=0,1 python init_and_serve.py
```

### Programmatic equivalent (what numa_autobind does)

```python
import os, ctypes

# 1. CPU affinity (kernel syscall, no libnuma needed)
os.sched_setaffinity(0, numa_local_cpus)

# 2. Memory policy via libnuma (needs libnuma.so.1)
libnuma = ctypes.CDLL("libnuma.so.1")
libnuma.numa_set_preferred(numa_node)  # soft — MPOL_PREFERRED
```

Step 1 works everywhere (pure syscall). Step 2 requires `numactl-libs` installed
in the container.

---

## Common Confusions

**"sched_setaffinity sets memory policy"** — No. It only controls CPU scheduling.
Memory policy is a separate kernel subsystem. They compose indirectly through
first-touch, but neither controls the other.

**"numa_run_on_node sets memory policy"** — No. It's a thin wrapper around
`sched_setaffinity`. Must pair with `numa_set_preferred` or `numa_set_membind`
for explicit memory policy.

**"First-touch is a memory policy"** — First-touch is the *behavior* of
MPOL_DEFAULT (no explicit policy). The page allocator places pages on the faulting
CPU's node. It's implicit, not a policy you set.

**"Pages migrate when you change CPU affinity"** — No. Existing physical pages stay
where they were allocated. Only new page faults respect the new CPU location. To
move existing pages, use `mbind` with `MPOL_MF_MOVE` or `migrate_pages(2)`.

---

## See Also

- [[ml-systems/hardware/numa-memory-architecture]] — NUMA topology, latency numbers, why placement matters
- [[ml-systems/hardware/pcie-dma-mechanics]] — PCIe DMA pipeline, why bulk transfers tolerate NUMA
- [[ml-systems/gpu/gpu-memory-hierarchy]] — GPU-side memory hierarchy
