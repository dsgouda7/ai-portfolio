"""
linux_benchmark.py — benchmark CodeTreeIndex against Linux drivers/net.

Downloads the drivers/net subtree from the Linux kernel, builds a
CodeTreeIndex, and evaluates 50 hand-crafted questions with ground-truth
file + line citations.

Usage
-----
    # Download source + build index + run eval (full pipeline, ~2-4 hours)
    python linux_benchmark.py all

    # Build only (saves index to benchmarks/data/linux-index/)
    python linux_benchmark.py build

    # Eval only (requires existing index)
    python linux_benchmark.py eval

    # Quick smoke test with 5 questions (no download needed if source present)
    python linux_benchmark.py smoke

Configuration is read from bench_config.yaml (tasks.code model).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_DIR  = Path(__file__).parent
_DATA_DIR   = _BENCH_DIR / "data"
_SRC_DIR    = _DATA_DIR / "linux_drivers_net"
_INDEX_DIR  = _DATA_DIR / "linux-index"
_Q_PATH     = _BENCH_DIR / "linux_questions.json"

# ── Ground-truth question bank ────────────────────────────────────────────────

# 50 questions across 5 categories (10 each).
# ground-truth: file is relative to drivers/net root; start_line is approximate
# (within 20 lines is considered correct for Recall@3 scoring).
QUESTIONS: list[dict] = [
    # ── Category 1: Behavior lookup ──────────────────────────────────────────
    {"id": "B01", "difficulty": "medium",
     "question": "Which function handles NAPI polling completion in the e1000 driver?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_clean",  "expected_start_line": 0},

    {"id": "B02", "difficulty": "medium",
     "question": "Which function resets the e1000 hardware adapter?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_reset",  "expected_start_line": 0},

    {"id": "B03", "difficulty": "hard",
     "question": "Which function in ixgbe handles MSI-X interrupt allocation?",
     "expected_file": "ethernet/intel/ixgbe/ixgbe_main.c",
     "expected_symbol": "ixgbe_acquire_msix_vectors",  "expected_start_line": 0},

    {"id": "B04", "difficulty": "easy",
     "question": "Where is the loopback interface transmit function?",
     "expected_file": "loopback.c",
     "expected_symbol": "loopback_xmit",  "expected_start_line": 0},

    {"id": "B05", "difficulty": "medium",
     "question": "Which function opens (brings up) the e1000e network device?",
     "expected_file": "ethernet/intel/e1000e/netdev.c",
     "expected_symbol": "e1000_open",  "expected_start_line": 0},

    {"id": "B06", "difficulty": "hard",
     "question": "Which function handles RX descriptor writeback in igb?",
     "expected_file": "ethernet/intel/igb/igb_main.c",
     "expected_symbol": "igb_clean_rx_irq",  "expected_start_line": 0},

    {"id": "B07", "difficulty": "medium",
     "question": "Which function configures multicast filtering in the e1000 driver?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_set_rx_mode",  "expected_start_line": 0},

    {"id": "B08", "difficulty": "easy",
     "question": "Which function is called when a tun/tap device is opened?",
     "expected_file": "tun.c",
     "expected_symbol": "tun_chr_open",  "expected_start_line": 0},

    {"id": "B09", "difficulty": "medium",
     "question": "Which function computes the checksum for transmitted packets in e1000?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_tx_csum",  "expected_start_line": 0},

    {"id": "B10", "difficulty": "hard",
     "question": "Where does the bonding driver select the active slave for TX?",
     "expected_file": "bonding/bond_main.c",
     "expected_symbol": "bond_start_xmit",  "expected_start_line": 0},

    # ── Category 2: Symbol definition ─────────────────────────────────────────
    {"id": "S01", "difficulty": "easy",
     "question": "Where is the e1000_ring struct defined?",
     "expected_file": "ethernet/intel/e1000e/e1000.h",
     "expected_symbol": "e1000_ring",  "expected_start_line": 0},

    {"id": "S02", "difficulty": "medium",
     "question": "Where is the ixgbe_adapter struct defined?",
     "expected_file": "ethernet/intel/ixgbe/ixgbe.h",
     "expected_symbol": "ixgbe_adapter",  "expected_start_line": 0},

    {"id": "S03", "difficulty": "easy",
     "question": "Where is the net_device_ops for the tun driver registered?",
     "expected_file": "tun.c",
     "expected_symbol": "tun_netdev_ops",  "expected_start_line": 0},

    {"id": "S04", "difficulty": "medium",
     "question": "Where is e1000_hw_stats defined?",
     "expected_file": "ethernet/intel/e1000/e1000_hw.h",
     "expected_symbol": "e1000_hw_stats",  "expected_start_line": 0},

    {"id": "S05", "difficulty": "hard",
     "question": "Where is the PHY operations struct for the igb driver?",
     "expected_file": "ethernet/intel/igb/e1000_phy.h",
     "expected_symbol": "e1000_phy_operations",  "expected_start_line": 0},

    {"id": "S06", "difficulty": "easy",
     "question": "Where is the veth_ops net_device_ops defined?",
     "expected_file": "veth.c",
     "expected_symbol": "veth_netdev_ops",  "expected_start_line": 0},

    {"id": "S07", "difficulty": "medium",
     "question": "Where is the 8139cp_private struct defined?",
     "expected_file": "ethernet/realtek/8139cp.c",
     "expected_symbol": "cp_private",  "expected_start_line": 0},

    {"id": "S08", "difficulty": "hard",
     "question": "Where is the bond_params struct defined in bonding?",
     "expected_file": "bonding/bond_options.h",
     "expected_symbol": "bond_params",  "expected_start_line": 0},

    {"id": "S09", "difficulty": "easy",
     "question": "Where is the dummy_ops net_device_ops for the dummy driver?",
     "expected_file": "dummy.c",
     "expected_symbol": "dummy_netdev_ops",  "expected_start_line": 0},

    {"id": "S10", "difficulty": "medium",
     "question": "Where is the macvlan_dev struct defined?",
     "expected_file": "macvlan.h",
     "expected_symbol": "macvlan_dev",  "expected_start_line": 0},

    # ── Category 3: Call-site search ──────────────────────────────────────────
    {"id": "C01", "difficulty": "medium",
     "question": "Which e1000 functions call netif_carrier_off?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_down",  "expected_start_line": 0},

    {"id": "C02", "difficulty": "hard",
     "question": "Where in the ixgbe driver is napi_schedule called?",
     "expected_file": "ethernet/intel/ixgbe/ixgbe_main.c",
     "expected_symbol": "ixgbe_msix_clean_rings",  "expected_start_line": 0},

    {"id": "C03", "difficulty": "easy",
     "question": "Which loopback function calls dev_kfree_skb?",
     "expected_file": "loopback.c",
     "expected_symbol": "loopback_xmit",  "expected_start_line": 0},

    {"id": "C04", "difficulty": "medium",
     "question": "Where does the e1000 driver call pci_enable_msi?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_request_irq",  "expected_start_line": 0},

    {"id": "C05", "difficulty": "hard",
     "question": "Which functions in igb call igb_alloc_rx_buffers?",
     "expected_file": "ethernet/intel/igb/igb_main.c",
     "expected_symbol": "igb_configure_rx_ring",  "expected_start_line": 0},

    {"id": "C06", "difficulty": "medium",
     "question": "Where in e1000e is the watchdog timer initialized?",
     "expected_file": "ethernet/intel/e1000e/netdev.c",
     "expected_symbol": "e1000_probe",  "expected_start_line": 0},

    {"id": "C07", "difficulty": "easy",
     "question": "Which tun function calls skb_reset_mac_header?",
     "expected_file": "tun.c",
     "expected_symbol": "tun_get_user",  "expected_start_line": 0},

    {"id": "C08", "difficulty": "hard",
     "question": "Where in the bonding driver is rtnl_lock acquired?",
     "expected_file": "bonding/bond_main.c",
     "expected_symbol": "bond_enslave",  "expected_start_line": 0},

    {"id": "C09", "difficulty": "medium",
     "question": "Which 8139cp function calls dma_alloc_coherent?",
     "expected_file": "ethernet/realtek/8139cp.c",
     "expected_symbol": "cp_init_rings",  "expected_start_line": 0},

    {"id": "C10", "difficulty": "easy",
     "question": "Where in veth is skb_pull called?",
     "expected_file": "veth.c",
     "expected_symbol": "veth_xmit",  "expected_start_line": 0},

    # ── Category 4: Bug-reproduction / narrow factual ────────────────────────
    {"id": "R01", "difficulty": "hard",
     "question": "Which e1000 function checks for and handles descriptor ring overflow?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_clean_tx_irq",  "expected_start_line": 0},

    {"id": "R02", "difficulty": "hard",
     "question": "Which igb function handles TXQ hung detection?",
     "expected_file": "ethernet/intel/igb/igb_main.c",
     "expected_symbol": "igb_watchdog_task",  "expected_start_line": 0},

    {"id": "R03", "difficulty": "medium",
     "question": "Which e1000 function recovers from a detected Tx unit hang?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_reset_task",  "expected_start_line": 0},

    {"id": "R04", "difficulty": "hard",
     "question": "Where does ixgbe handle a PCI bus error recovery?",
     "expected_file": "ethernet/intel/ixgbe/ixgbe_main.c",
     "expected_symbol": "ixgbe_io_error_detected",  "expected_start_line": 0},

    {"id": "R05", "difficulty": "medium",
     "question": "Which function validates e1000 EEPROM checksum?",
     "expected_file": "ethernet/intel/e1000/e1000_hw.c",
     "expected_symbol": "e1000_validate_eeprom_checksum",  "expected_start_line": 0},

    {"id": "R06", "difficulty": "easy",
     "question": "Which function in 8139cp checks for RX errors?",
     "expected_file": "ethernet/realtek/8139cp.c",
     "expected_symbol": "cp_rx_skb",  "expected_start_line": 0},

    {"id": "R07", "difficulty": "hard",
     "question": "Where does the bonding driver handle link failover?",
     "expected_file": "bonding/bond_main.c",
     "expected_symbol": "bond_miimon_commit",  "expected_start_line": 0},

    {"id": "R08", "difficulty": "medium",
     "question": "Which igb function handles unexpected MSI-X interrupts?",
     "expected_file": "ethernet/intel/igb/igb_main.c",
     "expected_symbol": "igb_msix_other",  "expected_start_line": 0},

    {"id": "R09", "difficulty": "easy",
     "question": "Which loopback function frees allocated memory on error?",
     "expected_file": "loopback.c",
     "expected_symbol": "loopback_dev_free",  "expected_start_line": 0},

    {"id": "R10", "difficulty": "hard",
     "question": "Where does e1000e handle ECC error correction?",
     "expected_file": "ethernet/intel/e1000e/netdev.c",
     "expected_symbol": "e1000e_check_ecc",  "expected_start_line": 0},

    # ── Category 5: Architecture ──────────────────────────────────────────────
    {"id": "A01", "difficulty": "medium",
     "question": "What is the interrupt handler entry point for the ixgbe driver?",
     "expected_file": "ethernet/intel/ixgbe/ixgbe_main.c",
     "expected_symbol": "ixgbe_intr",  "expected_start_line": 0},

    {"id": "A02", "difficulty": "easy",
     "question": "Which function is the PCI probe entry point for the e1000 driver?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_probe",  "expected_start_line": 0},

    {"id": "A03", "difficulty": "medium",
     "question": "How does the tun driver register itself as a character device?",
     "expected_file": "tun.c",
     "expected_symbol": "tun_init",  "expected_start_line": 0},

    {"id": "A04", "difficulty": "hard",
     "question": "Which function initializes the DMA rings in igb?",
     "expected_file": "ethernet/intel/igb/igb_main.c",
     "expected_symbol": "igb_setup_all_tx_resources",  "expected_start_line": 0},

    {"id": "A05", "difficulty": "medium",
     "question": "How does the e1000 driver handle power management suspend?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_suspend",  "expected_start_line": 0},

    {"id": "A06", "difficulty": "easy",
     "question": "What is the module init function for the bonding driver?",
     "expected_file": "bonding/bond_main.c",
     "expected_symbol": "bonding_init",  "expected_start_line": 0},

    {"id": "A07", "difficulty": "hard",
     "question": "Where does ixgbe set up flow control parameters?",
     "expected_file": "ethernet/intel/ixgbe/ixgbe_main.c",
     "expected_symbol": "ixgbe_fc_enable",  "expected_start_line": 0},

    {"id": "A08", "difficulty": "medium",
     "question": "Which function configures the e1000 to use jumbo frames?",
     "expected_file": "ethernet/intel/e1000/e1000_main.c",
     "expected_symbol": "e1000_change_mtu",  "expected_start_line": 0},

    {"id": "A09", "difficulty": "easy",
     "question": "What is the ethtool get_settings implementation for 8139cp?",
     "expected_file": "ethernet/realtek/8139cp.c",
     "expected_symbol": "cp_get_link_ksettings",  "expected_start_line": 0},

    {"id": "A10", "difficulty": "hard",
     "question": "How does the macvlan driver forward packets to the lower device?",
     "expected_file": "macvlan.c",
     "expected_symbol": "macvlan_start_xmit",  "expected_start_line": 0},
]


# ── Source download ───────────────────────────────────────────────────────────

def download_drivers_net(dest: Path, verbose: bool = True) -> None:
    """
    Clone only the drivers/net subtree from the Linux kernel repo using
    a sparse checkout (much faster than cloning the full kernel).
    """
    if dest.exists() and any(dest.iterdir()):
        if verbose:
            print(f"[download] drivers/net already present at {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[download] Sparse-cloning linux/drivers/net into {dest} ...")
        print("  This downloads ~200 MB. Run once; subsequent runs use the cache.")

    # git init + sparse checkout approach (no full clone needed)
    cmds = [
        ["git", "init", str(dest)],
        ["git", "-C", str(dest), "remote", "add", "origin",
         "https://github.com/torvalds/linux.git"],
        ["git", "-C", str(dest), "config", "core.sparseCheckout", "true"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True, capture_output=not verbose)

    sparse_file = dest / ".git" / "info" / "sparse-checkout"
    sparse_file.write_text("drivers/net/\n")

    subprocess.run(
        ["git", "-C", str(dest), "pull", "--depth=1", "origin", "master"],
        check=True,
        capture_output=not verbose,
    )
    if verbose:
        n = sum(1 for _ in dest.rglob("*.c")) + sum(1 for _ in dest.rglob("*.h"))
        print(f"[download] Done — {n} .c/.h files")


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_download(args: argparse.Namespace) -> None:
    download_drivers_net(Path(args.src) if hasattr(args, "src") else _SRC_DIR)


def cmd_build(args: argparse.Namespace) -> None:
    import code_benchmark as cb
    src = Path(getattr(args, "src", str(_SRC_DIR)))
    idx = Path(getattr(args, "index_dir", str(_INDEX_DIR)))
    if not src.exists():
        print(f"[build] Source not found: {src}  (run 'download' first)")
        sys.exit(1)

    ns = argparse.Namespace(
        src=src, index_dir=idx,
        provider="ollama", model="qwen2.5-coder:7b",
        embed_model="microsoft/codebert-base",
        cluster_size=4, min_lines=5, max_lines=300,
        exts=[".c", ".h"],
    )
    cb.cmd_build(ns)


def cmd_eval(args: argparse.Namespace) -> None:
    import code_benchmark as cb
    idx = Path(getattr(args, "index_dir", str(_INDEX_DIR)))

    # Write questions to file if not already present
    if not _Q_PATH.exists():
        _Q_PATH.write_text(json.dumps(QUESTIONS, indent=2), encoding="utf-8")
        print(f"[eval] Wrote {len(QUESTIONS)} questions to {_Q_PATH}")

    ns = argparse.Namespace(
        index_dir=idx,
        questions=_Q_PATH,
        reasoning_model=getattr(args, "reasoning_model", "qwen2.5-coder:7b"),
        top_k=2,
        limit=getattr(args, "limit", len(QUESTIONS)),
    )
    cb.cmd_eval(ns)


def cmd_smoke(args: argparse.Namespace) -> None:
    """Quick 5-question smoke test with the first 5 questions."""
    args.limit = 5
    cmd_eval(args)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="linux_benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="Sparse-clone drivers/net from linux/linux")
    sub.add_parser("build",    help="Build CodeTreeIndex from drivers/net")
    e = sub.add_parser("eval", help="Run 50-question evaluation")
    e.add_argument("--reasoning-model", default="qwen2.5-coder:7b", dest="reasoning_model")
    e.add_argument("--limit", default=50, type=int)
    sub.add_parser("smoke",    help="Quick 5-question smoke test")
    sub.add_parser("all",      help="download + build + eval")

    return p


def main() -> None:
    args = _parser().parse_args()
    if args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "all":
        cmd_download(args)
        cmd_build(args)
        cmd_eval(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
