# pip install requests pandas

import time
import requests
import pandas as pd

SEC_HEADERS = {
    # SEC asks you to identify yourself in User-Agent (name + contact).
    # Replace with your own.
    "User-Agent": "Dawid Kubicki dawid@pluscode.io",
    "Accept-Encoding": "gzip, deflate",
}

def cik10(cik: str) -> str:
    return str(cik).lstrip().lstrip("0").zfill(10)

def get_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def pick_latest_annual_usd(facts: dict, taxonomy: str, tag: str):
    """
    Pick the latest annual (10-K) USD fact for a tag if present.
    Uses 'USD' unit, filters for fp=='FY' and form=='10-K' where available.
    """
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag, {})
    units = node.get("units", {})
    rows = units.get("USD", [])  # common for US GAAP monetary values
    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Some rows may miss some fields; keep it tolerant.
    for col in ["end", "val", "fy", "fp", "form", "filed", "frame"]:
        if col not in df.columns:
            df[col] = None

    # Prefer annual 10-K FY datapoints.
    dfa = df.copy()
    dfa["end"] = pd.to_datetime(dfa["end"], errors="coerce")
    dfa["filed"] = pd.to_datetime(dfa["filed"], errors="coerce")

    dfa = dfa[(dfa["fp"] == "FY") & (dfa["form"] == "10-K")]
    if dfa.empty:
        # fallback: just take the latest by end date
        dfa = df.copy()
        dfa["end"] = pd.to_datetime(dfa["end"], errors="coerce")
        dfa["filed"] = pd.to_datetime(dfa["filed"], errors="coerce")

    dfa = dfa.sort_values(["end", "filed"], ascending=[False, False])
    row = dfa.iloc[0].to_dict()
    return row

def fundamentals_snapshot(cik: str) -> pd.DataFrame:
    facts = get_company_facts(cik)
    company = facts.get("entityName", "")
    tags = [
        ("Revenue", "us-gaap", "Revenues"),
        ("NetIncome", "us-gaap", "NetIncomeLoss"),
        ("TotalAssets", "us-gaap", "Assets"),
        ("TotalLiabilities", "us-gaap", "Liabilities"),
        ("OperatingCashFlow", "us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("SharesDiluted", "us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("EPSDiluted", "us-gaap", "EarningsPerShareDiluted"),
    ]

    out = []
    for label, tax, tag in tags:
        row = pick_latest_annual_usd(facts, tax, tag)
        out.append({
            "company": company,
            "cik": cik10(cik),
            "metric": label,
            "tag": f"{tax}:{tag}",
            "value": None if row is None else row.get("val"),
            "period_end": None if row is None else row.get("end"),
            "filed": None if row is None else row.get("filed"),
            "form": None if row is None else row.get("form"),
        })

    return pd.DataFrame(out)

if __name__ == "__main__":
    # Example: Apple CIK = 0000320193
    df = fundamentals_snapshot("0000320193")
    print(df)

    # If looping many companies, be polite with rate limiting.
    time.sleep(0.2)
