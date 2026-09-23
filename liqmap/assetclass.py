"""Sort a perp symbol into an asset class.

Hyperliquid lists well over a thousand markets across the canonical crypto
DEX and the HIP-3 builder DEXes, and the exchange does not tell you what kind
of thing any of them is. A flat alphabetical list of 1,393 tickers is not a
picker, it is a haystack.

HOW THE GUESSING WORKS, AND WHAT IT COSTS

Explicit lists first, pattern rules second, and crypto as the fallback only
for symbols on the canonical DEX -- where it is nearly always right, because
that DEX is crypto by definition. A HIP-3 symbol that matches nothing lands
in "other" rather than being guessed into a class, because a wrong label is
worse than an honest "not sure": it hides a market in a group you would never
look in.

This is a convenience for finding a market, not a data source. Nothing
downstream branches on the class, so a misfiled symbol costs you a moment of
searching and never a wrong number.
"""

from __future__ import annotations

from typing import Literal

Klass = Literal["crypto", "metals", "energy", "fx", "index", "equity",
                "etf", "other"]

ORDER: tuple[Klass, ...] = ("crypto", "metals", "energy", "fx", "index",
                            "equity", "etf", "other")

LABELS: dict[str, str] = {
    "crypto": "Crypto",
    "metals": "Metals",
    "energy": "Energy",
    "fx": "FX",
    "index": "Indices",
    "equity": "Stocks",
    "etf": "ETFs",
    "other": "Other",
}

METALS = {
    "GOLD", "XAU", "XAUUSD", "SILVER", "XAG", "XAGUSD", "PLATINUM", "XPT",
    "PALLADIUM", "XPD", "COPPER", "HG", "ALUMINIUM", "ALUMINUM", "ZINC",
    "NICKEL", "LITHIUM", "URANIUM", "STEEL", "IRON",
}

ENERGY = {
    "CL", "WTI", "CRUDE", "OIL", "BRENT", "BRENTOIL", "NATGAS", "NG",
    "GAS", "GASOLINE", "HEATINGOIL", "DIESEL", "COAL", "ETHANOL", "POWER",
    "ELECTRICITY",
}

# Majors as bare codes, plus the usual pair spellings.
FX_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "CNH", "CNY",
            "MXN", "BRL", "INR", "KRW", "TRY", "ZAR", "SEK", "NOK", "SGD",
            "HKD", "PLN", "DXY"}

INDICES = {
    "SPX", "SP500", "ES", "NDX", "NAS100", "NQ", "DJI", "DOW", "US30",
    "RUT", "RUSSELL", "RTY", "VIX", "JP225", "NIKKEI", "KR200", "KOSPI",
    "HSI", "DAX", "GER40", "FTSE", "UK100", "CAC", "EU50", "STOXX",
    "ASX", "AU200", "CHINA50", "TWSE", "SENSEX", "NIFTY", "DRAM",
}

# Recognisable fund/ETF names and the suffixes that mark leveraged products.
ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "ARKK", "GLD", "SLV", "USO",
    "TLT", "HYG", "EEM", "EFA", "XLF", "XLE", "XLK", "SMH", "SOXX", "SOXL",
    "TQQQ", "SQQQ", "UVXY", "EWJ", "EWY", "EWT", "EWZ", "KORU", "KSTR",
    "MAGS", "NCLD", "LYTE", "IBIT", "FBTC",
}
ETF_HINTS = ("BULL3X", "BEAR3X", "3X", "2X", "ETF")

# Crypto majors and the shapes crypto tickers take.
CRYPTO = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "MATIC",
    "POL", "LTC", "BCH", "TRX", "TON", "ATOM", "NEAR", "APT", "SUI", "SEI",
    "ARB", "OP", "INJ", "TIA", "PEPE", "WIF", "BONK", "SHIB", "HYPE", "PURR",
    "USDC", "USDT", "USDH", "DAI", "BNB", "ETC", "FIL", "ICP", "RUNE", "AAVE",
    "UNI", "MKR", "LDO", "CRV", "SNX", "COMP", "GMX", "DYDX", "JUP", "PYTH",
    "W", "ENA", "ETHFI", "EIGEN", "ZRO", "STRK", "BLUR", "BLAST", "MANTA",
    "ORDI", "SATS", "RATS", "MEME", "FLOKI", "BOME", "SLERF", "POPCAT",
    "MOG", "TURBO", "NEIRO", "GOAT", "PNUT", "ACT", "MOODENG", "ZEC", "XMR",
}
CRYPTO_PREFIXES = ("K", "1000", "10000")    # kPEPE, 1000SHIB and friends


def classify(symbol: str, dex: str = "") -> Klass:
    """Best guess at what kind of instrument this is.

    `dex` matters: the canonical (empty) DEX is crypto-only, so an unknown
    ticker there is safely crypto. An unknown ticker on a builder DEX could
    be anything, so it stays "other".
    """
    if not symbol:
        return "other"

    base = symbol.split(":")[-1].strip().upper()
    if not base:
        return "other"

    # Spot markets on Hyperliquid are indexed like @107.
    if base.startswith("@"):
        return "crypto"

    # A pair spelling settles it immediately.
    if "/" in base or "-" in base:
        left, _, right = base.replace("-", "/").partition("/")
        if left in FX_CODES and right in FX_CODES:
            return "fx"
        if left in CRYPTO or right in CRYPTO:
            return "crypto"

    core = base.replace("USD", "") if base not in CRYPTO and len(base) > 4 else base

    for table, klass in ((METALS, "metals"), (ENERGY, "energy"),
                         (INDICES, "index"), (ETFS, "etf")):
        if base in table or core in table:
            return klass                     # type: ignore[return-value]

    if base in FX_CODES or core in FX_CODES:
        return "fx"

    if base in CRYPTO or core in CRYPTO:
        return "crypto"

    if any(h in base for h in ETF_HINTS):
        return "etf"

    # kPEPE / 1000SHIB style wrappers around a known crypto ticker.
    for p in CRYPTO_PREFIXES:
        if base.startswith(p) and base[len(p):] in CRYPTO:
            return "crypto"

    if not dex:
        # Canonical Hyperliquid perps are crypto by construction.
        return "crypto"

    # A builder market with a short alphabetic ticker is almost always a
    # single stock -- that is what the equity DEXes list.
    if 1 <= len(base) <= 5 and base.isalpha():
        return "equity"

    return "other"


def label(klass: str) -> str:
    return LABELS.get(klass, klass.title())


def group(symbols: list[dict]) -> dict[str, list[dict]]:
    """Bucket `{"symbol", "dex", ...}` rows by class, in display order."""
    out: dict[str, list[dict]] = {k: [] for k in ORDER}
    for row in symbols:
        k = classify(row.get("symbol", ""), row.get("dex", ""))
        out[k].append(row)
    return {k: v for k, v in out.items() if v}
