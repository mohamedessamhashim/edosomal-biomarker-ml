"""
Map mouse gene symbols to human orthologs.

Primary:  Ensembl REST API (ortholog_one2one preferred)
Fallback: Hardcoded curated table (offline / API-down scenarios)

CRITICAL: Never blindly uppercase mouse symbols — many have different human names.
"""

import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    ENSEMBL_REST_SERVER, ENSEMBL_RATE_LIMIT_SLEEP,
    PROCESSED_DIR, EXTERNAL_DIR,
)

# ── Curated fallback table ─────────────────────────────────────────────────────
# Mouse → Human (VERIFIED — not simply uppercased)
HARDCODED_ORTHOLOGS: dict[str, str] = {
    # Known endosomal / retromer proteins
    "Aplp1":  "APLP1",  "Aplp2":  "APLP2",  "App":    "APP",
    "Chl1":   "CHL1",   "Mapt":   "MAPT",   "Sorl1":  "SORL1",
    "Sort1":  "SORT1",  "Ctsd":   "CTSD",   "Ctsb":   "CTSB",
    "Lamp1":  "LAMP1",  "Lamp2":  "LAMP2",  "Nrxn1":  "NRXN1",
    "Nrcam":  "NRCAM",  "L1cam":  "L1CAM",  "Ncam1":  "NCAM1",
    "Thy1":   "THY1",   "Cntn1":  "CNTN1",  "Nfasc":  "NFASC",
    "Cadm1":  "CADM1",  "Cadm4":  "CADM4",  "Negr1":  "NEGR1",
    "Opcml":  "OPCML",  "Ntm":    "NTM",    "Sez6l":  "SEZ6L",
    "Sez6":   "SEZ6",   "Clstn1": "CLSTN1", "Clstn3": "CLSTN3",
    "Bsg":    "BSG",    "Cd47":   "CD47",   "Itm2b":  "ITM2B",
    "Tubb3":  "TUBB3",  "Tubb2a": "TUBB2A", "Tubb4a": "TUBB4A",
    "Tubb4b": "TUBB4B", "Tuba1a": "TUBA1A", "Tuba1b": "TUBA1B",
    # Common exceptions where simple uppercase WOULD be wrong
    # (listed for documentation — add as needed)
    # "Trp53": "TP53",  (not in this dataset but kept as reminder)
}


# ── Ensembl REST API ───────────────────────────────────────────────────────────

def _ensembl_get(endpoint: str, params: dict | None = None, timeout: int = 10) -> dict | None:
    """Make a single GET request to Ensembl REST API, return parsed JSON or None."""
    url = f"{ENSEMBL_REST_SERVER}{endpoint}"
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(2)  # rate-limited
    except requests.exceptions.RequestException:
        pass
    return None


def map_mouse_to_human_ensembl(
    mouse_symbols: list[str],
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Query Ensembl for mouse → human orthology.

    Parameters
    ----------
    mouse_symbols : list of mouse gene symbols
    batch_size    : currently processes one-at-a-time (API limitation)

    Returns
    -------
    DataFrame with columns:
        mouse_symbol, human_symbol, human_ensembl_id,
        orthology_type, perc_id, confidence
    """
    results = []
    for symbol in mouse_symbols:
        endpoint = f"/homology/symbol/mus_musculus/{symbol}"
        data = _ensembl_get(endpoint, params={"type": "orthologues", "target_taxon": "9606"})
        if data:
            for entry in data.get("data", []):
                for h in entry.get("homologies", []):
                    if h.get("type") in ("ortholog_one2one", "ortholog_one2many"):
                        target = h.get("target", {})
                        results.append({
                            "mouse_symbol":    symbol,
                            "human_symbol":    target.get("gene_display_label", ""),
                            "human_ensembl_id": target.get("id", ""),
                            "orthology_type":  h.get("type", ""),
                            "perc_id":         target.get("perc_id", 0),
                            "confidence":      1 if h.get("type") == "ortholog_one2one" else 0,
                        })
                        break  # take first match
        time.sleep(ENSEMBL_RATE_LIMIT_SLEEP)

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["mouse_symbol", "human_symbol", "human_ensembl_id",
                 "orthology_type", "perc_id", "confidence"]
    )


# ── Combined mapper (hardcoded → API → uppercase fallback) ───────────────────

def map_orthologs(
    mouse_symbols: list[str],
    use_api: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Map mouse gene symbols to human orthologs.

    Strategy (in priority order):
    1. Hardcoded curated table
    2. Ensembl REST API (if use_api=True and symbol not in hardcoded table)
    3. Simple uppercase as last resort (flagged with confidence=0)

    Parameters
    ----------
    mouse_symbols : list of gene symbols to map
    use_api       : whether to query Ensembl REST
    cache_path    : if provided, load/save results here

    Returns
    -------
    DataFrame with columns:
        mouse_symbol, human_symbol, orthology_type, perc_id, confidence, source
    """
    # Load cache if available
    if cache_path and Path(cache_path).exists():
        cached = pd.read_csv(cache_path)
        already_mapped = set(cached["mouse_symbol"].tolist())
        to_map = [s for s in mouse_symbols if s not in already_mapped]
        print(f"[ortholog_mapping] Loaded {len(cached)} cached mappings; {len(to_map)} new")
    else:
        cached = pd.DataFrame()
        to_map = list(mouse_symbols)

    new_rows = []
    api_needed = []

    for sym in to_map:
        # Check hardcoded table (case-insensitive key search)
        match = next(
            (h for k, h in HARDCODED_ORTHOLOGS.items() if k.lower() == sym.lower()),
            None
        )
        if match:
            new_rows.append({
                "mouse_symbol":   sym,
                "human_symbol":   match,
                "orthology_type": "ortholog_one2one",
                "perc_id":        100,
                "confidence":     1,
                "source":         "hardcoded",
            })
        else:
            api_needed.append(sym)

    # Query API for the rest
    if use_api and api_needed:
        print(f"[ortholog_mapping] Querying Ensembl API for {len(api_needed)} symbols …")
        api_df = map_mouse_to_human_ensembl(api_needed)
        api_mapped = set(api_df["mouse_symbol"].tolist())
        api_df["source"] = "ensembl_api"
        for row in api_df.to_dict("records"):
            new_rows.append(row)

        # Uppercase fallback for anything that failed the API
        failed = [s for s in api_needed if s not in api_mapped]
        for sym in failed:
            new_rows.append({
                "mouse_symbol":   sym,
                "human_symbol":   sym.upper(),
                "orthology_type": "uppercase_fallback",
                "perc_id":        np.nan,
                "confidence":     0,
                "source":         "fallback",
            })
    elif api_needed:
        # No API — use uppercase fallback for everything not in hardcoded table
        for sym in api_needed:
            new_rows.append({
                "mouse_symbol":   sym,
                "human_symbol":   sym.upper(),
                "orthology_type": "uppercase_fallback",
                "perc_id":        np.nan,
                "confidence":     0,
                "source":         "fallback",
            })

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(
        columns=["mouse_symbol", "human_symbol", "orthology_type",
                 "perc_id", "confidence", "source"]
    )
    result = pd.concat([cached, new_df], ignore_index=True)

    # Save cache
    if cache_path:
        result.to_csv(cache_path, index=False)
        print(f"[ortholog_mapping] Saved mapping to {cache_path}")

    return result


# ── Summary stats ─────────────────────────────────────────────────────────────

def mapping_summary(mapping_df: pd.DataFrame) -> None:
    n = len(mapping_df)
    one2one  = (mapping_df["orthology_type"] == "ortholog_one2one").sum()
    hardcode = (mapping_df["source"] == "hardcoded").sum()
    api      = (mapping_df["source"] == "ensembl_api").sum()
    fallback = (mapping_df["source"] == "fallback").sum()
    print(f"  Total:       {n}")
    print(f"  1:1 orthologs: {one2one} ({one2one/n*100:.0f}%)")
    print(f"  Hardcoded:   {hardcode}")
    print(f"  Ensembl API: {api}")
    print(f"  Fallback:    {fallback}")


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_symbols = [
        "Aplp1", "Chl1", "Mapt", "App", "Aplp2",
        "Sort1", "Ctsd", "Sorl1", "Lamp1", "Cadm4",
    ]
    cache = EXTERNAL_DIR / "human_orthologs_cache.csv"
    result = map_orthologs(test_symbols, use_api=False, cache_path=cache)
    print("\nMapping results:")
    print(result.to_string(index=False))
    mapping_summary(result)
