"""Pool state off an Ethereum node, in one round trip.

Uses a BATCH JSON-RPC request rather than Multicall3. Both get every
pool in a single HTTP call; the batch needs no ABI encoding of a tuple
array, no deployed helper contract, and no web3 dependency -- each
entry is a bare four-byte selector with no arguments, which is all
`getReserves()`, `slot0()` and `liquidity()` take.

EVERY READ IS PINNED TO ONE BLOCK. The block number is fetched first
and passed as the block tag on every call in the batch, so the whole
set describes one instant. Reading each pool at "latest" would let a
new block land mid-batch and produce a book half from one block and
half from the next -- which is not a state the chain was ever in, and
the cycle it implies is not an arbitrage that existed.

The RPC URL is read from the environment, never logged, and never
returned in a payload.
"""

from __future__ import annotations

import os

import requests

from .dexarb import V2, V3, Pool, PoolState

SEL_RESERVES = "0x0902f1ac"      # getReserves()
SEL_SLOT0 = "0x3850c7bd"         # slot0()
SEL_LIQUIDITY = "0x1a686502"     # liquidity()

DEFAULT_TIMEOUT = 8.0


class FeedError(RuntimeError):
    """The node could not be reached or answered with an error."""


def rpc_url() -> str:
    return (os.environ.get("ETH_RPC_URL")
            or os.environ.get("RPC_URL") or "").strip()


def _word(data: str, i: int) -> int:
    """Word `i` of a hex return value, as an unsigned integer."""
    h = data[2:] if data.startswith("0x") else data
    lo = i * 64
    if len(h) < lo + 64:
        return 0
    return int(h[lo:lo + 64], 16)


def _post(url: str, payload, timeout: float):
    try:
        r = requests.post(url, json=payload, timeout=timeout,
                          headers={"Content-Type": "application/json"})
    except requests.RequestException as e:
        raise FeedError(f"rpc unreachable: {type(e).__name__}") from None
    if r.status_code != 200:
        raise FeedError(f"rpc returned {r.status_code}")
    try:
        return r.json()
    except ValueError:
        raise FeedError("rpc returned a non-JSON body") from None


def block_number(url: str, timeout: float = DEFAULT_TIMEOUT) -> int:
    out = _post(url, {"jsonrpc": "2.0", "id": 1,
                      "method": "eth_blockNumber", "params": []}, timeout)
    if not isinstance(out, dict) or "result" not in out:
        raise FeedError("no block number in the reply")
    return int(out["result"], 16)


def fetch_states(pools: list[Pool], url: str = "", block: int = 0,
                 timeout: float = DEFAULT_TIMEOUT) -> tuple[dict, int]:
    """Current state of every pool, and the block it was read at.

    Returns ({pool_key: PoolState}, block). A pool whose call failed is
    simply absent -- never present with zeroes, which would read as an
    empty book rather than as a missing one.
    """
    url = url or rpc_url()
    if not url:
        raise FeedError("no RPC URL set (ETH_RPC_URL or RPC_URL)")
    if not pools:
        return {}, 0

    if block <= 0:
        block = block_number(url, timeout)
    tag = hex(block)

    calls = []
    plan = []
    for p in pools:
        if p.kind == V2:
            calls.append((p.address, SEL_RESERVES))
            plan.append((p, "reserves"))
        else:
            calls.append((p.address, SEL_SLOT0))
            plan.append((p, "slot0"))
            calls.append((p.address, SEL_LIQUIDITY))
            plan.append((p, "liquidity"))

    payload = [{"jsonrpc": "2.0", "id": i,
                "method": "eth_call",
                "params": [{"to": to, "data": data}, tag]}
               for i, (to, data) in enumerate(calls)]

    out = _post(url, payload, timeout)
    if not isinstance(out, list):
        # A single error object comes back un-batched.
        msg = ""
        if isinstance(out, dict):
            msg = str(out.get("error", {}).get("message", ""))[:120]
        raise FeedError(f"batch rejected{': ' + msg if msg else ''}")

    by_id = {}
    for item in out:
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            by_id[item["id"]] = item

    states: dict[str, PoolState] = {}
    for i, (p, field) in enumerate(plan):
        item = by_id.get(i)
        if not item or "result" not in item:
            continue
        data = item["result"]
        if not isinstance(data, str) or len(data) < 66:
            continue
        st = states.get(p.key) or PoolState(block=block)
        if field == "reserves":
            st.reserve0 = _word(data, 0)
            st.reserve1 = _word(data, 1)
        elif field == "slot0":
            st.sqrt_price_x96 = _word(data, 0)
        else:
            st.liquidity = _word(data, 0)
        st.block = block
        states[p.key] = st

    # A pool that answered with zeroes is a pool with no book. Drop it
    # rather than pass it on: `DexArb` would quote it at nothing anyway,
    # but an entry that exists and is empty reads on the panel as a
    # live pool with no liquidity rather than as one we failed to read.
    return {k: v for k, v in states.items() if v.live}, block
