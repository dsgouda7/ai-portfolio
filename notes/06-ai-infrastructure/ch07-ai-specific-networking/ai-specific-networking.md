# Ch.7 — AI-Specific Networking

> **The story.** GPU-to-GPU communication became critical the moment models outgrew single-GPU memory. **InfiniBand** launched in **1999** as a high-performance computing interconnect for scientific clusters — offering 10 Gb/s when Ethernet topped out at 1 Gb/s. NVIDIA acquired **Mellanox** (the InfiniBand leader) in **April 2020** for $7 billion, making it the second-largest tech acquisition in Israeli history — a bet that AI training clusters would demand the same low-latency, high-bandwidth fabrics that had powered supercomputers for 20 years. But InfiniBand was expensive and server-grade. Consumer GPUs were bottlenecked by **PCIe 3.0** (16 GB/s per direction), which meant multi-GPU setups spent more time shuffling data between cards than computing. NVIDIA introduced **NVLink** with the **Pascal P100** in **April 2016** — a direct GPU-to-GPU interconnect running at 160 GB/s bidirectional (10× PCIe bandwidth). **NVLink 2.0** (V100, 2017) doubled it to 300 GB/s. **NVLink 3.0** (A100, 2020) reached 600 GB/s. **NVLink 4.0** (H100, 2022) hit **900 GB/s** — enabling tensor parallelism for 70B+ models that cannot fit on a single GPU. **Grace Hopper** (2024) integrates NVLink-C2C at **900 GB/s** directly between CPU and GPU, eliminating the PCIe bottleneck for CPU-GPU data transfer. Every layer of modern distributed AI training and inference — PyTorch DDP, DeepSpeed ZeRO, Megatron-LM tensor parallelism, vLLM disaggregated serving — is architected around these fabric primitives. You cannot understand why **Llama-2-70B** requires 4 GPUs with NVLink but struggles on 8 GPUs with PCIe without understanding the bandwidth hierarchy. You cannot debug why a distributed training job's `all-reduce` is taking 2 seconds per step instead of 200 ms without understanding NCCL's topology detection and ring algorithms.
>
> **Where you are in the curriculum.** This is Chapter 7 of the AI Infrastructure track. **Running scenario:** InferenceBase has successfully deployed Llama-3-8B on a single RTX 4090 (Ch.1-6), cutting monthly costs from $80k to $1,095. Traffic is growing — the CEO wants to scale to 40,000 req/day and support Llama-2-70B for enterprise customers. A 70B model (140 GB in FP16) cannot fit on a single GPU. The Platform Engineer must now understand GPU-to-GPU networking to enable tensor parallelism across 4 GPUs.
> **Notation.** `NVLink` = NVIDIA's proprietary GPU-to-GPU interconnect (900 GB/s on H100). `InfiniBand` = High-performance cluster fabric (200 Gb/s = 25 GB/s per lane). `RDMA` = Remote Direct Memory Access (zero-copy network transfers, bypassing CPU). `PCIe` = PCI Express bus (16 GB/s per direction on Gen4 ×16). `NCCL` = NVIDIA Collective Communications Library (optimized all-reduce, all-gather primitives). `Tensor Parallelism` = Splitting one model's layers across multiple GPUs. `All-Reduce` = Synchronization primitive where all GPUs exchange and sum gradients.
<!-- notation: key variables defined here -->

---

## 0 · The Challenge — Where We Are

> 🎬 **Animation placeholder** — Deterministic animation showing GPU communication patterns: Single-GPU baseline → Multi-GPU with PCIe (bottleneck highlighted) → Multi-GPU with NVLink (bandwidth improvement shown). Visual will demonstrate 70B model split across 4 GPUs with attention layer sharded across devices, showing token-by-token inference with cross-GPU communication. Compare PCIe (16 GB/s → 8.75s to transfer 140GB) vs NVLink (900 GB/s → 0.16s). Target: 30-second loop showing communication timeline with bandwidth comparison overlay.

---

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
>
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ **Ch.1 (GPU Architecture)**: RTX 4090 identified as target (24GB VRAM, 1.0 TB/s bandwidth, $1.50/hr)
- ✅ **Ch.2 (Memory Budgets)**: Llama-3-8B FP16 fits in 20GB (16GB params + 4GB KV cache)
- ✅ **Ch.3 (Quantization)**: INT4 quantization → 8GB params, enables multi-batch inference
- ✅ **Ch.4 (Distributed Training)**: Understand data/tensor/pipeline parallelism strategies
- ✅ **Ch.5 (Inference Optimization)**: PagedAttention + continuous batching → 1.2s p95 latency, 12k req/day ✅
- ✅ **Ch.6 (Serving Frameworks)**: vLLM deployed, achieving throughput and latency targets ✅
- 📊 **Current state**:
  - **Production**: 1× RTX 4090, Llama-3-8B INT4, vLLM serving
  - **Cost**: $1,095/month (93% under $15k budget) ✅
  - **Performance**: 1.2s p95 latency, 12,000 req/day (120% of target) ✅
  - **Quality**: 96.2% accuracy (above 95% target) ✅

**What's blocking us**:

🚨 **New requirement: Enterprise customers want Llama-2-70B — 140GB model won't fit on 24GB GPU**

**Current situation**: CEO's Q1 planning meeting

```
CEO: "Great work on the cost savings. Now I need you to plan for Q1 growth.
     Enterprise customers are asking for Llama-2-70B support — they need
     higher reasoning quality for legal document analysis. Our current 8B
     model isn't cutting it for complex contracts."

Engineer: "70B model is 140 GB in FP16. Even with INT4 quantization, that's
          35 GB. Our RTX 4090 has 24 GB VRAM. It won't fit on one GPU."

CEO: "So we need multiple GPUs?"

Engineer: "Yes, at least 4 GPUs. But there's a problem: if I connect 4 GPUs
          via PCIe, the cross-GPU communication is so slow that inference
          latency explodes. PCIe is only 16 GB/s — transferring 35 GB of
          activations between GPUs would take 2+ seconds *per layer*."

CEO: "What's the solution?"

Engineer: "NVLink. It's 900 GB/s — 56× faster than PCIe. But NVLink is only
          available on datacenter GPUs like A100 or H100, not consumer cards.
          We'd need to switch from RTX 4090 ($1.50/hr) to A100 ($2.50/hr).
          And I need to understand the networking topology to architect this
          correctly — NVSwitch vs direct NVLink, InfiniBand for multi-node,
          RDMA for low-latency communication."

CEO: "Will this break our $15k/month budget?"

Engineer: "4× A100 @ $2.50/hr = $10/hr = $7,300/month. Still under budget,
          but we need to validate the architecture first. I have 2 weeks to:
          1. Understand NVLink vs PCIe bandwidth trade-offs
          2. Profile Llama-2-70B tensor parallelism with NCCL
          3. Measure actual latency on multi-GPU setup
          4. Design InfiniBand topology if we need >8 GPUs in the future"
```

**Problems**:
1. ❌ **PCIe bottleneck**: PCIe Gen4 ×16 = 16 GB/s (one direction). Llama-2-70B INT4 = 35 GB parameters. Multi-GPU inference requires transferring activation tensors between GPUs every layer (hundreds of MB per forward pass). At 16 GB/s, cross-GPU communication dominates latency → 5-10s per inference (vs 2s target) ⚠️
2. ❌ **No NVLink on consumer GPUs**: RTX 4090 has no NVLink — only PCIe. To get NVLink, must switch to datacenter GPUs (A100, H100) at higher cost ($2.50-$4/hr vs $1.50/hr)
3. ❌ **Topology complexity**: 4 GPUs in a server can connect via:
   - Direct NVLink mesh (A100: 12 NVLink lanes, can connect 8 GPUs fully)
   - NVSwitch fabric (H100: 18 NVLinks + NVSwitch for full bisection bandwidth)
   - PCIe + CPU relay (consumer GPUs: all traffic through PCIe → CPU bottleneck)
   - InfiniBand for multi-node (200 Gb/s = 25 GB/s, but requires RDMA setup)

   **Team has zero expertise in HPC networking** — don't know how to choose topology or debug NCCL communication failures
4. ❌ **Unknown latency impact**: Tensor parallelism splits each transformer layer across GPUs. Every `matmul` requires an `all-reduce` collective to synchronize results. If `all-reduce` takes 10ms (slow PCIe) vs 1ms (fast NVLink), and Llama-2-70B has 80 layers, that's **800ms vs 80ms communication overhead per inference** — determines whether we meet the 2s latency target or not
5. ❌ **NCCL tuning required**: NCCL (NVIDIA Collective Communications Library) auto-detects topology and selects ring/tree algorithms. But default settings may choose suboptimal paths (e.g., going through PCIe when NVLink is available). Must understand NCCL topology tables and tuning parameters (`NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`, etc.) to debug performance
6. ❌ **InfiniBand unknown**: For future scale-out (16+ GPUs across 2-4 nodes), will need InfiniBand fabric. Team doesn't know:
   - InfiniBand vs Ethernet (ROCE)
   - RDMA setup (GPUDirect RDMA to bypass CPU)
   - InfiniBand switch topology (fat-tree, dragonfly)
   - Cost implications (Mellanox switches, cables, NICs)

**Business impact**:
- **$1M+ enterprise deals blocked**: Cannot serve 70B models → losing high-value customers to competitors
- **Latency regression risk**: Naive multi-GPU setup could push p95 latency to 5-10s (vs 2s target), violating SLAs
- **Over-provisioning risk**: Buying 4× H100 ($4/hr each = $11,680/month) when 4× A100 ($2.50/hr = $7,300/month) would suffice → burning $4,380/month unnecessarily
- **Architecture lock-in**: Choosing wrong topology (e.g., PCIe-only servers) makes future scale-out to 16-32 GPUs impossible without full hardware replacement
- CEO: "We need 70B support in Q1. Figure out the networking. If we can't hit <2s latency with multi-GPU, we're back to OpenAI's API."

**What this chapter unlocks**:

🚀 **GPU-to-GPU networking fundamentals for multi-GPU inference and training**:
1. **Understand interconnect hierarchy**: PCIe vs NVLink vs InfiniBand → bandwidth, latency, topology
2. **Profile communication patterns**: Tensor parallelism for 70B model → measure `all-reduce` overhead with NCCL
3. **Read topology tables**: `nvidia-smi topo -m` → identify NVLink bridges, PCIe switches, NUMA domains
4. **Design multi-GPU architecture**: 4-GPU NVLink mesh (A100) vs 8-GPU NVSwitch (H100) vs multi-node InfiniBand (16+ GPUs)
5. **Validate latency**: Llama-2-70B inference on 4× A100 with NVLink → measure actual p95 latency < 2s ✅

⚡ **Expected outcomes**:
- **Architecture decision**: 4× A100 (80GB) with NVLink identified as target (vs RTX 4090 PCIe-only or H100 overkill)
- **Cost estimate**:
  ```
  4× A100 80GB @ $2.50/hr × 730 hr/month = $7,300/month
  vs $15,000 budget → $7,700 headroom (51% under budget) ✅
  vs $80,000 OpenAI baseline → $72,700 savings/month (91% reduction) ✅
  ```
- **Bandwidth verification**:
  ```
  PCIe Gen4 ×16: 16 GB/s (one direction) → 35 GB transfer = 2.2s → FAILS latency target ❌
  NVLink 3.0 (A100): 600 GB/s bidirectional = 300 GB/s per direction → 35 GB = 0.12s → PASSES ✅
  InfiniBand HDR: 200 Gb/s = 25 GB/s → 35 GB = 1.4s → marginal (okay for multi-node, slower than NVLink)
  ```
- **Tensor parallelism validated**: Llama-2-70B INT4 (35 GB) split across 4× A100 (80 GB each) → 8.75 GB per GPU + 4 GB KV cache = ~13 GB used per GPU (16% of 80 GB VRAM) ✅
- **Latency benchmark**:
  ```
  Llama-2-70B INT4 on 4× A100 with NVLink:
    - Compute per layer: ~50ms (dominated by matmul in INT4)
    - Communication per layer (all-reduce): ~1.5ms (NVLink + NCCL)
    - 80 layers × 51.5ms = 4.1s total
    - With batching + optimization → ~1.8s p95 latency (under 2s target) ✅

  Same setup with PCIe (no NVLink):
    - Communication per layer: ~15ms (10× slower)
    - 80 layers × (50 + 15)ms = 5.2s → FAILS ❌
  ```
- **Next question unlocked**: "How do we deploy and monitor this in production?" → Need Ch.8-10 for cloud infrastructure, MLOps, production platform

**Constraint status after Ch.7**:
- #1 (Cost): ⚡ **ON TRACK** ($7,300/month for 4× A100, under $15k budget) ✅
- #2 (Latency): ⚡ **VALIDATED** (1.8s p95 with NVLink, under 2s target) ✅
- #3 (Throughput): ⚡ **EXCEEDS TARGET** (4× A100 can handle 40,000+ req/day with parallel serving) ✅
- #4 (Memory): ⚡ **RESOLVED** (70B INT4 fits across 4× 80GB GPUs with headroom) ✅
- #5 (Quality): ⚡ **ASSUMED** (Llama-2-70B benchmarks show >97% accuracy on complex tasks) ✅
- #6 (Reliability): ⚡ **UNKNOWN** (need Ch.9-10 for fault tolerance, monitoring, auto-scaling)

**Foundation established**: Understand GPU networking → can now design cloud deployment and production monitoring (Ch.8-10)

---

## Animation

<!-- TODO: add animation GIF here -->

---

## 1 · Core Idea

GPU-to-GPU communication is the hidden multiplier on all distributed AI workloads. A transformer layer forward pass might take 10 ms to compute — but if shuffling the activations between GPUs takes 20 ms, your system is spending 67% of its time moving data, not computing. The interconnect hierarchy — PCIe (16 GB/s), NVLink (900 GB/s), InfiniBand (25 GB/s per lane) — determines whether multi-GPU inference is viable or a latency disaster. **Every tensor parallelism strategy, every distributed training algorithm, every disaggregated serving architecture is designed around the assumption that GPUs can exchange large tensors in <1 ms.** Remove that assumption (use PCIe instead of NVLink) and the entire software stack falls apart.

---

## 2 · Running Example

InferenceBase's Llama-2-70B model has 70 billion parameters (140 GB in FP16, 35 GB in INT4). This cannot fit on a single GPU (even H100 has "only" 80 GB VRAM). The Platform Engineer must split it across 4 GPUs using tensor parallelism — each GPU holds 1/4 of each weight matrix (8.75 GB per GPU). Every forward pass requires cross-GPU communication: after each GPU computes its slice of the matrix multiply, an `all-reduce` collective synchronizes the results. With NVLink (600-900 GB/s), this `all-reduce` takes ~1-2 ms. With PCIe (16 GB/s), it takes ~15-20 ms. Over 80 transformer layers, that difference is 80-160 ms (NVLink) vs 1,200-1,600 ms (PCIe) — determining whether inference latency is 1.8s or 5.2s.

---

## 3 · The Interconnect Hierarchy

### PCIe — The CPU-Centric Bottleneck

**PCI Express** is the standard bus connecting GPUs to the CPU and to each other. PCIe Gen4 ×16 provides **16 GB/s** in each direction (32 GB/s bidirectional). PCIe Gen5 doubles this to 32 GB/s per direction (64 GB/s bidirectional).

```
PCIe Topology (Typical Consumer/Server Setup)

         CPU
          │
    ┌─────┴─────┐
    │  PCIe    │
    │  Switch  │  (Gen4 ×16 lanes)
    └─┬───┬───┬┘
      │   │   │
    GPU0 GPU1 GPU2

GPU0 ↔ GPU1 communication:
  GPU0 → PCIe → CPU → PCIe → GPU1
  Bandwidth: 16 GB/s (one direction)
  Latency: ~5-10 μs (includes CPU relay overhead)
```

**The problem**: All GPU-to-GPU traffic flows through the CPU's PCIe root complex. This means:
1. **Limited bandwidth**: 16 GB/s is 50× slower than GPU's internal memory bandwidth (1 TB/s)
2. **CPU bottleneck**: If CPU is busy, PCIe transfers queue up
3. **NUMA issues**: Multi-socket servers have PCIe switches on each CPU — cross-socket GPU traffic has even worse latency

**When PCIe is acceptable**:
- Data parallelism (each GPU has a full model copy — only gradients need syncing, not activations)
- Small models (<7B parameters)
- Low communication frequency (e.g., gradient sync once per batch, not once per layer)

**When PCIe fails**:
- Tensor parallelism (activations transferred every layer → 80+ times per forward pass)
- Pipeline parallelism with fine-grained stages (frequent cross-GPU activation passing)
- Large models (70B+) where activation tensors are hundreds of MB

### NVLink — The Direct GPU-to-GPU Highway

**NVLink** is NVIDIA's proprietary high-speed interconnect that bypasses the CPU entirely. Each NVLink "lane" is a point-to-point bidirectional link between two GPUs.

```
NVLink Topology (A100 8-GPU Server)

    GPU0 ←NVLink→ GPU1 ←NVLink→ GPU2 ←NVLink→ GPU3
     ↕              ↕              ↕              ↕
    GPU4 ←NVLink→ GPU5 ←NVLink→ GPU6 ←NVLink→ GPU7

Each GPU has 12 NVLink lanes (A100 3.0):
  - Direct connections to 6 neighbors
  - 600 GB/s total bidirectional bandwidth per GPU
  - Any-to-any communication via multi-hop routing
```

**NVLink Generations**:

| Generation | GPU | Bandwidth per Lane | Lanes per GPU | Total per GPU |
|------------|-----|-------------------|---------------|---------------|
| NVLink 1.0 | P100 | 20 GB/s | 4-6 | 80-120 GB/s |
| NVLink 2.0 | V100 | 25 GB/s | 6 | 150 GB/s |
| NVLink 3.0 | A100 | 25 GB/s | 12 | 300 GB/s (600 GB/s bidir) |
| NVLink 4.0 | H100 | 50 GB/s | 18 | 900 GB/s (bidir) |

**Key advantages**:
1. **50-100× faster than PCIe**: 600-900 GB/s vs 16 GB/s
2. **Sub-microsecond latency**: Direct GPU-to-GPU, no CPU in the path
3. **Scalable topology**: 8 GPUs can fully connect via NVLink mesh (A100) or NVSwitch (H100)
4. **RDMA-capable**: GPU memory is directly accessible from remote GPUs (zero-copy transfers)

**NVSwitch** (H100, Grace Hopper):
- Crossbar switch connecting all GPUs with full bisection bandwidth
- 18 NVLink 4.0 ports per GPU → all connected through NVSwitch
- Enables **900 GB/s × 8 GPUs = 7.2 TB/s aggregate bandwidth** in a single server
- Eliminates multi-hop routing — every GPU-to-GPU pair has direct NVSwitch path

**Limitations**:
- **Datacenter GPUs only**: NVLink is not available on consumer GPUs (RTX 3090, RTX 4090)
- **Single-node only**: NVLink connects GPUs in the same server (up to 8 GPUs with NVSwitch)
- **Cost premium**: A100/H100 with NVLink cost 2-3× more than consumer GPUs

### InfiniBand — The Multi-Node Fabric

**InfiniBand** is a high-performance cluster interconnect designed for HPC. It provides RDMA (Remote Direct Memory Access) — network transfers that bypass the CPU and kernel, delivering near-zero latency.

```
InfiniBand Topology (Multi-Node Cluster)

   ┌─────────────────────────────────────────────┐
   │         InfiniBand Switch Fabric             │
   │              (Fat-Tree Topology)             │
   └─┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬┘
     │     │     │     │     │     │     │     │
   Node0 Node1 Node2 Node3 Node4 Node5 Node6 Node7

Each node:
  - 8× A100 GPUs (NVLink-connected within node)
  - 1-4× InfiniBand NICs (200 Gb/s each)
  - GPUDirect RDMA enabled (GPU memory → IB NIC → network, zero-copy)
```

**InfiniBand Speeds** (per lane):

| Standard | Speed | Bandwidth | Year |
|----------|-------|-----------|------|
| FDR | 56 Gb/s | 7 GB/s | 2011 |
| EDR | 100 Gb/s | 12.5 GB/s | 2014 |
| HDR | 200 Gb/s | 25 GB/s | 2017 |
| NDR | 400 Gb/s | 50 GB/s | 2020 |

**Key features**:
1. **RDMA**: Zero-copy network transfers, <1 μs latency (vs 10-50 μs for TCP/IP)
2. **GPUDirect RDMA**: GPU memory directly accessible over InfiniBand (no CPU copy)
3. **Low latency**: ~1-2 μs node-to-node (vs 50-100 μs for Ethernet)
4. **High bandwidth**: 200-400 Gb/s = 25-50 GB/s (competitive with NVLink for cross-node)
5. **Scalable fabric**: Thousands of nodes in a single non-blocking fat-tree network

**When InfiniBand is required**:
- Multi-node training (>8 GPUs across 2+ servers)
- Distributed inference (disaggregated KV cache across nodes)
- Large-scale clusters (16-128+ GPUs)

**Alternatives to InfiniBand**:
- **RoCE** (RDMA over Converged Ethernet): RDMA on standard Ethernet hardware (cheaper, but higher latency ~5-10 μs)
- **AWS EFA** (Elastic Fabric Adapter): AWS's proprietary RDMA fabric (similar to InfiniBand, cloud-native)
- **GCP GPUDirect**: Google Cloud's GPU-to-GPU fabric (custom RDMA implementation)

**Cost implications**:
- InfiniBand switch (36-port HDR): ~$15,000-$30,000
- InfiniBand NIC (200 Gb/s): ~$1,000-$2,000 per server
- InfiniBand cables: ~$100-$500 each (active optical cables for long distances)
- **Total cluster networking cost**: ~$50k-$200k for 8-32 node setup (vs $5k-$20k for Ethernet)

---

## 4 · Communication Patterns in Distributed AI

### Collective Operations

Distributed training and inference rely on **collective communication primitives** where all GPUs participate in a synchronized data exchange.

**All-Reduce**: Sum tensors across all GPUs, return result to all GPUs

```
All-Reduce Example (4 GPUs, gradient synchronization)

Before:
  GPU0: [1, 2, 3]
  GPU1: [4, 5, 6]
  GPU2: [7, 8, 9]
  GPU3: [10, 11, 12]

After all-reduce (sum):
  GPU0: [22, 26, 30]  (sum of all)
  GPU1: [22, 26, 30]
  GPU2: [22, 26, 30]
  GPU3: [22, 26, 30]
```

**Used for**:
- Gradient synchronization in data parallelism (each GPU has a local gradient, must average)
- Tensor parallelism (each GPU has partial output, must sum)
- ZeRO optimizer state sharding (gather full optimizer state for update)

**All-Gather**: Gather tensors from all GPUs, concatenate, return full tensor to all GPUs

```
All-Gather Example (4 GPUs, embedding lookup)

Before:
  GPU0: [1, 2]
  GPU1: [3, 4]
  GPU2: [5, 6]
  GPU3: [7, 8]

After all-gather:
  GPU0: [1, 2, 3, 4, 5, 6, 7, 8]
  GPU1: [1, 2, 3, 4, 5, 6, 7, 8]
  GPU2: [1, 2, 3, 4, 5, 6, 7, 8]
  GPU3: [1, 2, 3, 4, 5, 6, 7, 8]
```

**Used for**:
- Gathering sharded model weights (e.g., ZeRO-3)
- Assembling full activation tensor in tensor parallelism
- Combining predictions from multiple GPUs

**Reduce-Scatter**: Sum tensors, scatter results (each GPU gets a slice)

```
Reduce-Scatter Example (4 GPUs)

Before:
  GPU0: [1, 2, 3, 4]
  GPU1: [5, 6, 7, 8]
  GPU2: [9, 10, 11, 12]
  GPU3: [13, 14, 15, 16]

After reduce-scatter (sum, then split):
  GPU0: [28]  (1+5+9+13)
  GPU1: [32]  (2+6+10+14)
  GPU2: [36]  (3+7+11+15)
  GPU3: [40]  (4+8+12+16)
```

**Used for**:
- ZeRO-2/ZeRO-3 optimizer state distribution
- Efficient gradient averaging (combine reduce-scatter + all-gather)

**Broadcast**: Send tensor from one GPU to all others

```
Broadcast Example

Before:
  GPU0: [1, 2, 3, 4]  (source)
  GPU1: [0, 0, 0, 0]
  GPU2: [0, 0, 0, 0]
  GPU3: [0, 0, 0, 0]

After broadcast:
  GPU0: [1, 2, 3, 4]
  GPU1: [1, 2, 3, 4]
  GPU2: [1, 2, 3, 4]
  GPU3: [1, 2, 3, 4]
```

**Used for**:
- Distributing model weights from rank-0 at initialization
- Broadcasting hyperparameters or control signals

### Ring All-Reduce Algorithm

NCCL (NVIDIA Collective Communications Library) implements collectives using a **ring algorithm** that maximizes bandwidth utilization.

```
Ring All-Reduce (4 GPUs)

Step 1: Reduce-Scatter Phase
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0 (ring)
  Each GPU sends 1/N of its data to the next GPU
  After N-1 steps, each GPU has the sum of 1/N of all data

Step 2: All-Gather Phase
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0 (ring)
  Each GPU forwards its summed chunk to the next GPU
  After N-1 steps, all GPUs have the full result

Total steps: 2(N-1) = 6 steps for N=4 GPUs
Data transferred per GPU: 2(N-1)/N = 1.5× the data size
```

**Why ring is optimal**:
- Every GPU sends and receives simultaneously (full-duplex bandwidth utilization)
- No single GPU is a bottleneck (unlike tree-based algorithms where root GPU is overloaded)
- Scales to any number of GPUs (linear time, not logarithmic — but maximizes bandwidth)

**NVLink + Ring All-Reduce**:
- With NVLink mesh, NCCL automatically detects topology and selects ring path
- With NVSwitch, NCCL can use multiple parallel rings (8-ring all-reduce on 8 GPUs)
- Achieves **near-theoretical bandwidth**: 0.9-0.95× peak NVLink bandwidth

**PCIe + Ring All-Reduce**:
- Ring must traverse CPU PCIe switch multiple times
- Bandwidth limited to PCIe speed (16 GB/s), not GPU internal bandwidth (1 TB/s)
- Achieves only **0.3-0.5× PCIe bandwidth** due to CPU contention

---

## 5 · Comparison Table — PCIe vs NVLink vs InfiniBand

| Metric | PCIe Gen4 ×16 | NVLink 3.0 (A100) | NVLink 4.0 (H100) | InfiniBand HDR | InfiniBand NDR |
|--------|---------------|-------------------|-------------------|----------------|----------------|
| **Bandwidth (one direction)** | 16 GB/s | 300 GB/s | 450 GB/s | 25 GB/s | 50 GB/s |
| **Bidirectional** | 32 GB/s | 600 GB/s | 900 GB/s | 50 GB/s | 100 GB/s |
| **Latency** | 5-10 μs | <1 μs | <1 μs | 1-2 μs | 1-2 μs |
| **Scope** | Single node (CPU-centric) | Single node (GPU-to-GPU) | Single node (GPU-to-GPU) | Multi-node | Multi-node |
| **Max GPUs (single fabric)** | 4-8 (via PCIe switch) | 8 (NVLink mesh) | 8 (NVSwitch) | Thousands (fabric) | Thousands (fabric) |
| **Cost (per GPU)** | Included (standard PCIe) | $2-3k premium vs PCIe-only | $5-8k premium vs PCIe-only | ~$1-2k (NIC cost) | ~$2-3k (NIC cost) |
| **RDMA Support** | No | Yes (GPUDirect P2P) | Yes (GPUDirect P2P) | Yes (GPUDirect RDMA) | Yes (GPUDirect RDMA) |
| **Best Use Case** | Consumer GPUs, data parallelism | Tensor parallelism, single-node training | Largest models (175B+), single-node | Multi-node training, clusters | Largest clusters (128+ GPUs) |
| **Typical Cloud Availability** | All instances | A100, H100 instances | H100 instances | AWS p4d, Azure ND-series | AWS p5, Azure NDv5 |

**When to choose each**:

| Your requirement | Recommendation | Rationale |
|-----------------|----------------|-----------|
| Single GPU, <10B params | PCIe-only (RTX 4090) | No multi-GPU needed, PCIe irrelevant |
| 2-4 GPUs, data parallelism | PCIe-only (RTX 4090) | Gradient sync is infrequent, PCIe acceptable |
| 2-4 GPUs, tensor parallelism <30B | NVLink (A100 or A10G with NVLink) | Moderate cross-GPU traffic, NVLink helps |
| 4-8 GPUs, tensor parallelism 30-70B | NVLink 3.0 (A100) | High cross-GPU traffic, NVLink required |
| 4-8 GPUs, tensor parallelism 70-175B | NVLink 4.0 (H100) + NVSwitch | Extreme cross-GPU bandwidth needs |
| 8-32 GPUs, multi-node training | InfiniBand HDR + NVLink | Within-node NVLink, cross-node InfiniBand |
| 32-128+ GPUs, large clusters | InfiniBand NDR + NVLink | Supercomputer-scale, need fastest fabric |

---

## 6 · Tensor Parallelism Communication Pattern

Tensor parallelism (also called *intra-layer model parallelism*) splits each weight matrix across multiple GPUs. This requires communication *every layer*, making interconnect speed critical.

**Example: Transformer Feed-Forward Layer (Llama-2-70B)**

```
Standard (Single GPU):
  Input: [batch, seq_len, hidden_dim] = [1, 2048, 8192]
  Weight1: [hidden_dim, ffn_dim] = [8192, 28672] (235 MB in FP16)
  Intermediate: matmul(Input, Weight1) = [1, 2048, 28672]
  Activation: GELU(Intermediate)
  Weight2: [ffn_dim, hidden_dim] = [28672, 8192] (235 MB)
  Output: matmul(Activation, Weight2) = [1, 2048, 8192]

Tensor Parallelism (4 GPUs):
  Split Weight1 column-wise: each GPU holds [8192, 7168] (59 MB)

  GPU0: matmul(Input, Weight1[:,0:7168]) → [1, 2048, 7168]
  GPU1: matmul(Input, Weight1[:,7168:14336]) → [1, 2048, 7168]
  GPU2: matmul(Input, Weight1[:,14336:21504]) → [1, 2048, 7168]
  GPU3: matmul(Input, Weight1[:,21504:28672]) → [1, 2048, 7168]

  → All-Gather: Concatenate outputs → [1, 2048, 28672] on all GPUs
  → GELU activation (independent per GPU)

  Split Weight2 row-wise: each GPU holds [7168, 8192] (59 MB)

  GPU0: matmul(Activation[:,0:7168], Weight2[0:7168,:]) → [1, 2048, 8192]
  GPU1: matmul(Activation[:,7168:14336], Weight2[7168:14336,:]) → [1, 2048, 8192]
  GPU2: matmul(Activation[:,14336:21504], Weight2[14336:21504,:]) → [1, 2048, 8192]
  GPU3: matmul(Activation[:,21504:28672], Weight2[21504:28672,:]) → [1, 2048, 8192]

  → All-Reduce (sum): Add outputs → [1, 2048, 8192] on all GPUs
```

**Communication volume per layer**:
- All-Gather: Transfer 7168 values × 2 bytes × 3 other GPUs = ~42 MB
- All-Reduce: Transfer 8192 values × 2 bytes × 3 other GPUs = ~48 MB
- **Total**: ~90 MB per feed-forward layer

**Latency calculation**:

| Interconnect | Bandwidth | Time per layer | 80 layers (total) |
|--------------|-----------|---------------|-------------------|
| PCIe Gen4 ×16 | 16 GB/s | 90 MB / 16 GB/s = **5.6 ms** | 5.6 ms × 80 = **448 ms** |
| NVLink 3.0 (A100) | 300 GB/s | 90 MB / 300 GB/s = **0.3 ms** | 0.3 ms × 80 = **24 ms** |
| NVLink 4.0 (H100) | 900 GB/s | 90 MB / 900 GB/s = **0.1 ms** | 0.1 ms × 80 = **8 ms** |
| InfiniBand HDR | 25 GB/s | 90 MB / 25 GB/s = **3.6 ms** | 3.6 ms × 80 = **288 ms** |

**Conclusion**: NVLink reduces communication overhead by **18× (A100) to 56× (H100)** compared to PCIe. InfiniBand is acceptable for multi-node (when NVLink isn't an option), but still 12× slower than H100 NVLink.

---

## 7 · Reading GPU Topology with `nvidia-smi`

NVIDIA's `nvidia-smi topo -m` command displays the GPU interconnect topology, showing which GPUs are connected via NVLink, PCIe, or CPU relay.

**Example output (4× A100 with NVLink)**:

```
$ nvidia-smi topo -m

        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    NV12    NV12    0-63           0
GPU1    NV12     X      NV12    NV12    0-63           0
GPU2    NV12    NV12     X      NV12    0-63           0
GPU3    NV12    NV12    NV12     X      0-63           0

Legend:
  X    = Self
  NV#  = Connection traversing a bonded set of # NVLinks
  PIX  = Connection traversing PCIe and X NVLinks
  PHB  = Connection traversing PCIe Host Bridge
  SYS  = Connection traversing PCIe + NUMA socket
```

**Interpretation**:
- `NV12` = Direct NVLink connection with 12 lanes (A100 3.0 → 600 GB/s bidirectional)
- All GPUs are on the same NUMA node (socket 0)
- No PCIe traversal between GPUs (optimal topology)

**Example output (4× RTX 4090 with PCIe only)**:

```
$ nvidia-smi topo -m

        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      PHB     PHB     SYS     0-31           0
GPU1    PHB      X      PHB     SYS     0-31           0
GPU2    PHB     PHB      X      SYS     0-31           0
GPU3    SYS     SYS     SYS      X      32-63          1

Legend:
  PHB  = Connection traversing PCIe Host Bridge (same CPU socket)
  SYS  = Connection traversing PCIe + NUMA socket (cross-socket)
```

**Interpretation**:
- `PHB` = PCIe connection through CPU host bridge (16 GB/s)
- `SYS` = Cross-socket connection (GPU3 is on a different CPU socket → even slower)
- GPU3 on different NUMA node → cross-NUMA traffic adds latency
- **Problem**: GPU0 ↔ GPU3 must traverse PCIe + QPI/UPI cross-socket link → 2× slower than GPU0 ↔ GPU1

**What to look for**:
- ✅ **Best**: All `NV#` entries → full NVLink mesh
- ⚠️ **Okay**: `PHB` within same NUMA node → PCIe but no cross-socket penalty
- ❌ **Bad**: `SYS` entries → cross-socket traffic, high latency
- ❌ **Worst**: `PIX` entries → PCIe + partial NVLink (asymmetric topology, NCCL struggles to optimize)

**NCCL Tuning**:
- NCCL reads `nvidia-smi topo` output to auto-detect topology
- Set `NCCL_DEBUG=INFO` to see topology detection logs
- Set `NCCL_P2P_DISABLE=1` to force all traffic through CPU (debug tool — never use in production)
- Set `NCCL_IB_DISABLE=1` to disable InfiniBand (useful for debugging IB fabric issues)

---

## Key Diagrams

<!-- TODO: add key diagrams -->

---

## 8 · What Can Go Wrong

### 1. NVLink Not Detected — NCCL Falls Back to PCIe

**Symptom**: Multi-GPU training or inference is 10-50× slower than expected. `nvidia-smi topo -m` shows `NV#` links, but NCCL reports PCIe bandwidth in logs.

**Root cause**: Driver mismatch, NVLink bridge not seated correctly, or SBIOS setting disabled NVLink.

**Debug**:
```bash
# Check NVLink status
nvidia-smi nvlink --status

# Expected output (working NVLink):
GPU 0: 12 NVLinks, Link 0: UP, Link 1: UP, ...

# If all links show "DOWN", NVLink is disabled
```

**Fix**:
- Update NVIDIA driver to latest version
- Check SBIOS settings (some servers disable NVLink by default)
- Reseat GPU NVLink bridge cables (physical connection issue)
- Verify with `nvidia-smi nvlink -i 0 -c 0` (counter should increment when GPUs communicate)

### 2. Cross-Socket NUMA Bottleneck

**Symptom**: 2 GPUs on the same node are 2-3× slower than expected, even with NVLink.

**Root cause**: GPUs are on different CPU sockets (NUMA nodes). Cross-socket QPI/UPI link (50-100 GB/s) is slower than intra-socket communication.

**Debug**:
```bash
nvidia-smi topo -m | grep SYS
# If any "SYS" entries appear, you have cross-socket traffic
```

**Fix**:
- Pin GPUs to CPUs on the same socket using `numactl`:
  ```bash
  numactl --cpunodebind=0 --membind=0 python train.py --local_rank 0
  numactl --cpunodebind=0 --membind=0 python train.py --local_rank 1
  ```
- Or configure PyTorch DDP to use same-socket GPUs:
  ```python
  import torch.distributed as dist
  dist.init_process_group(backend='nccl')
  # Set CUDA_VISIBLE_DEVICES to only same-socket GPUs
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # GPUs on socket 0
  ```

### 3. InfiniBand Fabric Misconfiguration

**Symptom**: Multi-node training works but is slower than single-node. Cross-node all-reduce takes 100-500 ms instead of expected 5-10 ms.

**Root cause**: InfiniBand fabric not configured, GPUDirect RDMA disabled, or using RoCE instead of native InfiniBand.

**Debug**:
```bash
# Check InfiniBand status
ibstat

# Expected output:
CA 'mlx5_0'
  Port 1:
    State: Active
    Physical state: LinkUp
    Rate: 200 Gb/s (HDR)
    Link layer: InfiniBand

# Check if GPUDirect RDMA is enabled
cat /proc/driver/nvidia/gpudirect-rdma/version
# Should show "nvidia-peermem" module loaded
```

**Fix**:
- Install MLNX_OFED drivers (Mellanox OpenFabrics Enterprise Distribution):
  ```bash
  wget https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.9-0.5.6.0/...
  sudo ./mlnxofedinstall --add-kernel-support --skip-repo
  ```
- Enable GPUDirect RDMA:
  ```bash
  sudo modprobe nvidia-peermem
  lsmod | grep nvidia_peermem  # Verify module loaded
  ```
- Set NCCL environment variables:
  ```bash
  export NCCL_IB_DISABLE=0  # Enable InfiniBand
  export NCCL_IB_HCA=mlx5_0  # Specify IB adapter
  export NCCL_IB_GID_INDEX=3  # RoCE mode (if using Ethernet)
  export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
  ```

### 4. PCIe Gen3 Instead of Gen4

**Symptom**: Bandwidth is only 8 GB/s instead of expected 16 GB/s.

**Root cause**: GPU or motherboard negotiated PCIe Gen3 (8 GB/s per direction) instead of Gen4 (16 GB/s).

**Debug**:
```bash
nvidia-smi -q | grep "Bus Id\|Link Speed"

# Expected output (Gen4):
    Bus Id                          : 00000000:17:00.0
    Link Speed                      : 16 GT/s  (Gen4)

# If shows 8 GT/s, you're running Gen3 → half bandwidth!
```

**Fix**:
- Enable PCIe Gen4 in SBIOS (some systems default to Gen3 for stability)
- Check if GPU is in a Gen4 PCIe slot (some motherboards have mix of Gen3/Gen4 slots)
- Update motherboard firmware to latest version (fixes PCIe negotiation bugs)

### 5. NVLink Bandwidth Not Saturated

**Symptom**: NCCL reports NVLink detected, but bandwidth is only 50-100 GB/s instead of 600 GB/s.

**Root cause**: Small message sizes, CPU-side bottleneck, or NCCL not using optimal algorithm.

**Debug**:
```bash
# Run NCCL bandwidth test
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 4

# Expected output (4× A100 with NVLink):
#   Avg bus bandwidth: 480-550 GB/s (0.8-0.9× theoretical max)

# If seeing <100 GB/s, NCCL is not using NVLink correctly
```

**Fix**:
- Increase message size (NCCL needs >1 MB messages to saturate NVLink)
- Set `NCCL_DEBUG=INFO` and check for warnings about topology
- Update NCCL to latest version (older versions had NVLink detection bugs)
- Disable P2P fallback: `NCCL_P2P_LEVEL=NVL` (force NVLink, fail if unavailable)

---

## 9 · Progress Check

You should now be able to:

1. **Explain the bandwidth hierarchy**: PCIe (16 GB/s) → NVLink (600-900 GB/s) → InfiniBand (25-50 GB/s). Understand when each is the bottleneck.

2. **Calculate communication overhead**: Given a model size (e.g., Llama-2-70B = 140 GB FP16), number of GPUs (4), and tensor parallelism strategy, compute the all-reduce volume per layer and multiply by interconnect latency to estimate communication cost.

3. **Read `nvidia-smi topo -m`**: Identify NVLink vs PCIe vs cross-socket connections. Recognize when topology is suboptimal (PHB, SYS, PIX entries).

4. **Choose the right topology**:
   - 1-4 GPUs, <30B model → PCIe acceptable for data parallelism
   - 4-8 GPUs, 30-70B model → NVLink required for tensor parallelism
   - 8+ GPUs, multi-node → InfiniBand + NVLink (within-node NVLink, cross-node InfiniBand)

5. **Debug communication slowdowns**: Use NCCL bandwidth tests, check NVLink status with `nvidia-smi nvlink`, verify NUMA affinity with `numactl`, validate InfiniBand with `ibstat`.

6. **Estimate cost**:
   - 4× A100 80GB (NVLink) @ $2.50/hr = $7,300/month
   - 8× H100 80GB (NVSwitch) @ $4.00/hr = $23,360/month
   - Add InfiniBand cost for multi-node: ~$1,000-$2,000 per server (NIC) + ~$15,000-$30,000 (switch)

---

## Where This Reappears

<!-- TODO: add forward pointer table -->

---

## 10 · Bridge to Next Chapter

We now understand GPU networking — NVLink for single-node tensor parallelism, InfiniBand for multi-node scale-out. InferenceBase has validated that Llama-2-70B on 4× A100 with NVLink can hit <2s latency at $7,300/month (under the $15k budget, 91% savings vs $80k OpenAI baseline).

**But a new question emerges**: Where do we deploy these 4× A100 GPUs? On-premises bare metal? AWS p4d instances? Azure ND-series? Google Cloud A2 instances? RunPod? Lambda Labs? Each has different:
- **Pricing models**: On-demand vs reserved vs spot instances
- **Network topology**: Some providers have NVLink + InfiniBand (AWS p4d), others have PCIe-only (AWS g5)
- **Availability**: H100 instances have 6-12 month waitlists, A100 available immediately
- **Cost multipliers**: Azure charges 30% more than AWS for same hardware, but has better egress pricing
- **Orchestration**: Kubernetes vs SLURM vs custom batch system

**The next bottleneck**: **Cloud infrastructure cost modeling and vendor selection**

**Next chapter** (Ch.8 — Cloud Infrastructure): Learn how to compare cloud GPU providers (AWS vs GCP vs Azure vs bare-metal), model true cost (compute + storage + egress), choose instance types (on-demand vs spot vs reserved), and design the deployment architecture (single-node vs multi-node, auto-scaling, fault tolerance). Build a cost calculator that takes model size, traffic volume, and latency requirements as input, and outputs the optimal cloud deployment strategy.

**The question Ch.8 will answer**: "Given InferenceBase's requirements (Llama-2-70B, 40k req/day, <2s latency, $15k/month budget), which cloud provider and instance type should we use, and how do we architect it for cost and reliability?"

---

## See Also

- **Prerequisite**: [Ch.1 GPU Architecture](../ch01_gpu_architecture/gpu-architecture.md) — Understand GPU hardware before networking
- **Prerequisite**: [Ch.4 Parallelism & Distributed Training](../ch04_parallelism_and_distributed_training/parallelism-and-distributed-training.md) — Tensor parallelism strategies
- **Next**: [Ch.8 Cloud Infrastructure] *(planned)* — Deploy multi-GPU setup in cloud, cost modeling
- **Related**: [DevOps Ch.7 Networking Basics](../../devops_fundamentals/networking/README.md) — TCP/IP, load balancing (prerequisite)
- **Tools**: [NCCL Tests](https://github.com/NVIDIA/nccl-tests) — Benchmark collective communication bandwidth
- **Tools**: [nvidia-smi topology](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) — Inspect GPU interconnect
- **Paper**: [Megatron-LM: Training Multi-Billion Parameter Models](https://arxiv.org/abs/1909.08053) — Tensor parallelism patterns
- **Paper**: [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965) — Pipeline parallelism communication patterns
