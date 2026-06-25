#!/usr/bin/env python3
"""
Buffett-style equity analysis report from free public market data.

Usage:
    python buffett_report.py AAPL
    python buffett_report.py D05.SI
    python buffett_report.py MSFT --output report.md

Dependencies (install once):
    pip install yfinance

No API key required. Data via Yahoo Finance (unofficial).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

RISK_FREE_RATE = 0.021  # 10Y SGS / US proxy; override with --rf
USER_AGENT = "buffett-report/1.0"


@dataclass
class Metrics:
    ticker: str
    name: str
    sector: str
    industry: str
    currency: str
    price: float | None
    market_cap: float | None
    enterprise_value: float | None
    trailing_pe: float | None
    forward_pe: float | None
    price_to_book: float | None
    dividend_yield: float | None
    beta: float | None
    roe: float | None
    roic: float | None
    profit_margin: float | None
    operating_margin: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    total_debt: float | None
    total_cash: float | None
    ebitda: float | None
    interest_coverage: float | None
    cash_conversion: float | None
    owner_earnings: float | None
    owner_earnings_yield: float | None
    fcf: float | None
    ann_return_5y: float | None
    ann_vol_5y: float | None
    sharpe_5y: float | None
  # 8-question filter
    q_circle: bool = False
    q_durability: bool = False
    q_moat: bool = False
    q_pricing: bool = False
    q_earnings_quality: bool = False
    q_debt_safety: bool = False
    q_integrity: bool = True  # cannot verify from data; assume yes unless red flags
    q_price: bool = False
    red_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _sf(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not t:
        raise ValueError("Ticker cannot be empty")
    # Common Singapore shorthand: D05 -> D05.SI
    if t.isalpha() or (len(t) <= 4 and t[-1].isdigit() and "." not in t):
        if len(t) <= 3 and t.isalpha():
            pass  # US tickers like AAPL
        elif len(t) <= 4 and any(c.isdigit() for c in t) and not t.endswith(".SI"):
            return f"{t}.SI"
    return t


def _yahoo_chart_metrics(ticker: str) -> tuple[float | None, float | None, float | None]:
    """Fallback 5Y return/vol/Sharpe via Yahoo chart API (stdlib only)."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        f"?range=5y&interval=1mo"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        closes = payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
        closes = [c for c in closes if c is not None]
        if len(closes) < 12:
            return None, None, None
        rets = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        std = math.sqrt(var)
        ann_ret = (1 + mean) ** 12 - 1
        ann_vol = std * math.sqrt(12)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else None
        return ann_ret, ann_vol, sharpe
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return None, None, None


# urllib.parse used in chart fallback


def fetch_with_yfinance(ticker: str) -> Metrics:
    assert yf is not None
    t = yf.Ticker(ticker)
    info: dict[str, Any] = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    if not info.get("symbol") and not info.get("shortName"):
        raise ValueError(f"No data found for ticker '{ticker}'. Check symbol (e.g. D05.SI, AAPL).")

    # Financial statements (most recent annual column)
    def latest_col(df):
        if df is None or df.empty:
            return {}
        col = df.columns[0]
        return {str(k): _sf(df.loc[k, col]) for k in df.index if k in df.index}

    inc = {}
    bal = {}
    cf = {}
    try:
        inc = latest_col(t.financials)
        bal = latest_col(t.balance_sheet)
        cf = latest_col(t.cashflow)
    except Exception:
        pass

    net_income = inc.get("Net Income") or _sf(info.get("netIncomeToCommon"))
    op_cf = cf.get("Operating Cash Flow") or cf.get("Total Cash From Operating Activities")
    capex = cf.get("Capital Expenditure")
    if capex is not None:
        capex = abs(capex)
    da = cf.get("Depreciation") or inc.get("Reconciled Depreciation") or inc.get("Depreciation And Amortization")
    ebit = inc.get("EBIT") or inc.get("Operating Income")
    interest = inc.get("Interest Expense")
    if interest is not None:
        interest = abs(interest)
    revenue = inc.get("Total Revenue")
    equity = bal.get("Stockholders Equity") or bal.get("Total Stockholder Equity")
    total_debt = bal.get("Total Debt") or bal.get("Long Term Debt")
    cash = bal.get("Cash And Cash Equivalents") or bal.get("Cash")
    ebitda = inc.get("EBITDA") or _sf(info.get("ebitda"))

    # Owner earnings (heuristic maintenance capex = min(capex, D&A) or 80% of D&A)
    owner_earnings = None
    if net_income is not None:
        maint = None
        if capex is not None and da is not None:
            maint = min(capex, da) if da > 0 else capex * 0.7
        elif da is not None:
            maint = da * 0.8
        elif capex is not None:
            maint = capex * 0.7
        if maint is not None:
            owner_earnings = net_income + (da or 0) - maint

    fcf = None
    if op_cf is not None and capex is not None:
        fcf = op_cf - capex

    cash_conversion = None
    if op_cf is not None and net_income and net_income != 0:
        cash_conversion = op_cf / net_income

    roe = _sf(info.get("returnOnEquity"))
    if roe is not None and abs(roe) < 5:
        roe = roe * 100  # yfinance sometimes returns decimal

    roic = None
    tax_rate = 0.17
    if ebit is not None and equity is not None:
        nopat = ebit * (1 - tax_rate)
        invested = equity + (total_debt or 0) - (cash or 0)
        if invested and invested > 0:
            roic = (nopat / invested) * 100

    interest_coverage = None
    if ebit is not None and interest and interest > 0:
        interest_coverage = ebit / interest

    debt_to_equity = _sf(info.get("debtToEquity"))
    if debt_to_equity is not None and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100

    mcap = _sf(info.get("marketCap"))
    oe_yield = None
    if owner_earnings and mcap and mcap > 0:
        oe_yield = owner_earnings / mcap

    ann_ret, ann_vol, sharpe = None, None, None
    try:
        hist = t.history(period="5y", interval="1mo")
        if len(hist) >= 12:
            rets = hist["Close"].pct_change().dropna()
            mean = rets.mean()
            std = rets.std()
            ann_ret = (1 + mean) ** 12 - 1
            ann_vol = std * math.sqrt(12)
            if ann_vol and ann_vol > 0:
                sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    except Exception:
        ann_ret, ann_vol, sharpe = _yahoo_chart_metrics(ticker)

    div_yield = _sf(info.get("dividendYield"))
    if div_yield is not None and div_yield < 1:
        div_yield = div_yield * 100

    m = Metrics(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector") or "Unknown",
        industry=info.get("industry") or "Unknown",
        currency=info.get("currency") or "USD",
        price=_sf(info.get("currentPrice") or info.get("regularMarketPrice")),
        market_cap=mcap,
        enterprise_value=_sf(info.get("enterpriseValue")),
        trailing_pe=_sf(info.get("trailingPE")),
        forward_pe=_sf(info.get("forwardPE")),
        price_to_book=_sf(info.get("priceToBook")),
        dividend_yield=div_yield,
        beta=_sf(info.get("beta")),
        roe=roe,
        roic=roic,
        profit_margin=_sf(info.get("profitMargins")),
        operating_margin=_sf(info.get("operatingMargins")),
        revenue_growth=_sf(info.get("revenueGrowth")),
        earnings_growth=_sf(info.get("earningsGrowth")),
        debt_to_equity=debt_to_equity,
        current_ratio=_sf(info.get("currentRatio")),
        total_debt=total_debt,
        total_cash=cash,
        ebitda=ebitda,
        interest_coverage=interest_coverage,
        cash_conversion=cash_conversion,
        owner_earnings=owner_earnings,
        owner_earnings_yield=oe_yield,
        fcf=fcf,
        ann_return_5y=ann_ret,
        ann_vol_5y=ann_vol,
        sharpe_5y=sharpe,
    )

    if revenue is None and not inc:
        m.warnings.append("Limited financial statement data; metrics may be incomplete.")
    return m


def apply_quick_filter(m: Metrics) -> Metrics:
    """Heuristic 8-question Buffett quick filter from available data."""
    # Q1 Circle of competence — data cannot judge; true if we have business description
    m.q_circle = m.sector != "Unknown" and m.industry != "Unknown"

    # Q2 Durability — profitable with positive margins or large cap
    m.q_durability = (m.profit_margin or 0) > 0 or (m.market_cap or 0) > 5e9

    # Q3 Moat — ROIC/ROE thresholds
    m.q_moat = (m.roic or 0) >= 12 or (m.roe or 0) >= 15

    # Q4 Pricing power — margins + revenue growth
    m.q_pricing = (m.operating_margin or 0) >= 0.15 or (m.profit_margin or 0) >= 0.10

    # Q5 Earnings quality
    m.q_earnings_quality = (m.cash_conversion or 0) >= 0.8 or (
        m.cash_conversion is None and (m.fcf or 0) > 0
    )

    # Q6 Debt safety
    safe_de = (m.debt_to_equity or 0) < 100
    safe_cov = (m.interest_coverage or 99) >= 3
    m.q_debt_safety = safe_de and safe_cov
    if (m.debt_to_equity or 0) > 150:
        m.red_flags.append("High leverage (D/E > 1.5x)")

    # Q7 Integrity — no automated test; flag if extreme accounting mismatch
    if m.cash_conversion is not None and m.cash_conversion < 0.5 and (m.profit_margin or 0) > 0:
        m.red_flags.append("Low cash conversion vs reported earnings — verify accounting quality")
        m.q_integrity = False
    else:
        m.q_integrity = True

    # Q8 Reasonable price — owner earnings yield or PE vs growth
    if m.owner_earnings_yield and m.owner_earnings_yield >= 0.05:
        m.q_price = True
    elif m.trailing_pe and 0 < m.trailing_pe < 25:
        m.q_price = True
    elif m.trailing_pe and m.trailing_pe > 40:
        m.red_flags.append("Elevated trailing P/E — limited margin of safety")

    no_count = sum(
        1
        for q in (
            m.q_circle,
            m.q_durability,
            m.q_moat,
            m.q_pricing,
            m.q_earnings_quality,
            m.q_debt_safety,
            m.q_integrity,
            m.q_price,
        )
        if not q
    )
    if no_count >= 4:
        m.red_flags.append(f"Quick filter: {no_count}/8 No answers — consider passing")
    return m


def fmt_pct(value: float | None, decimals: int = 1, already_percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if already_percent:
        return f"{value:.{decimals}f}%"
    return f"{value * 100:.{decimals}f}%"


def fmt_money(value: float | None, currency: str = "USD") -> str:
    if value is None:
        return "N/A"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= div:
            return f"{currency} {value / div:.2f}{unit}"
    return f"{currency} {value:,.0f}"


def fmt_ratio(value: float | None, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def moat_assessment(m: Metrics) -> str:
    roic = m.roic or 0
    roe = m.roe or 0
    if roic >= 20 or roe >= 20:
        strength, trend = "strong", "stable"
    elif roic >= 12 or roe >= 15:
        strength, trend = "medium", "stable"
    else:
        strength, trend = "weak", "narrowing"
    moat_type = "intangible / scale"
    if "bank" in m.industry.lower() or "financial" in m.sector.lower():
        moat_type = "switching costs + regulatory scale"
    elif "tech" in m.sector.lower() or "software" in m.industry.lower():
        moat_type = "network effects / intangible assets"
    elif "consumer" in m.sector.lower():
        moat_type = "brand / intangible assets"
    return f"{moat_type} + {strength} + {trend}"


def conclusion(m: Metrics) -> str:
    if not m.q_integrity:
        return "Don't Buy — integrity / earnings quality concerns require manual verification"
    nos = sum(
        1
        for q in (
            m.q_moat,
            m.q_earnings_quality,
            m.q_debt_safety,
            m.q_price,
            m.q_durability,
        )
        if not q
    )
    if nos >= 3:
        return "Keep Watching — fails multiple quality or valuation gates; not enough margin of safety"
    if m.q_moat and m.q_price and m.q_earnings_quality:
        return "Buy / Hold — passes core quality and valuation screens; verify circle of competence manually"
    return "Keep Watching — mixed signals; deeper research required before committing capital"


def margin_of_safety_note(m: Metrics) -> str:
    if m.owner_earnings_yield and m.owner_earnings_yield >= 0.07:
        return "~20–30% (high certainty tier) — owner earnings yield ≥7%"
    if m.trailing_pe and m.trailing_pe < 18:
        return "~30–40% (generally excellent tier) — reasonable multiple"
    if m.trailing_pe and m.trailing_pe > 35:
        return "<10% (low) — paying premium price; requires exceptional growth"
    return "~30–40% (uncertainty factors present) — verify with manual DCF"


def generate_report(m: Metrics) -> str:
    m = apply_quick_filter(m)
    yes_no = lambda b: "Yes" if b else "No"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H UTC")

    lines = [
        f"# Buffett Analysis Report: {m.name} ({m.ticker})",
        f"",
        f"*Generated {now} · Data: Yahoo Finance (free, unofficial) · No API key*",
        f"",
        f"---",
        f"",
        f"## Conclusion",
        f"**{conclusion(m)}**",
        f"",
        f"## Circle of Competence Assessment",
        f"**{'Inside circle (sector known)' if m.q_circle else 'Boundary / outside — verify manually'}**",
        f"- Sector: {m.sector} · Industry: {m.industry}",
        f"- Can you explain in one paragraph how this business makes money? **You must answer this yourself.**",
        f"",
        f"## Key Assumptions",
        f"1. Reported financials are materially accurate (automated check only).",
        f"2. Maintenance capex ≈ min(CapEx, D&A) or 80% of D&A for owner earnings estimate.",
        f"3. Risk-free rate = {RISK_FREE_RATE*100:.1f}% for Sharpe calculation.",
        f"4. 10-year competitive position does not deteriorate structurally.",
        f"5. Current market price reflects Mr. Market's mood, not necessarily intrinsic value.",
        f"",
        f"## Quick Filter (8 Questions)",
        f"",
        f"| # | Dimension | Result |",
        f"|---|-----------|--------|",
        f"| 1 | Circle of Competence | {yes_no(m.q_circle)} (data partial) |",
        f"| 2 | Durability (10Y) | {yes_no(m.q_durability)} |",
        f"| 3 | Moat | {yes_no(m.q_moat)} |",
        f"| 4 | Pricing Power | {yes_no(m.q_pricing)} |",
        f"| 5 | Earnings Quality | {yes_no(m.q_earnings_quality)} |",
        f"| 6 | Debt Safety | {yes_no(m.q_debt_safety)} |",
        f"| 7 | Management Integrity | {yes_no(m.q_integrity)} (manual verify) |",
        f"| 8 | Reasonable Price | {yes_no(m.q_price)} |",
        f"",
        f"## Business Quality",
        f"- **Moat:** {moat_assessment(m)}",
        f"- **Management:** integrity unverified / capital allocation review required / owner mentality unknown",
        f"- **Business model:** {'franchise' if m.q_moat else 'commodity / hybrid'}",
        f"- **Institutional imperative warning:** absent (not detectable from data)",
        f"",
        f"## Financial Snapshot",
        f"- **Price:** {fmt_money(m.price, m.currency)}",
        f"- **Market cap:** {fmt_money(m.market_cap, m.currency)}",
        f"- **ROE:** {fmt_ratio(m.roe, '%', 1)}",
        f"- **ROIC (est.):** {fmt_ratio(m.roic, '%', 1)}",
        f"- **Profit margin:** {fmt_pct(m.profit_margin)}",
        f"- **Operating margin:** {fmt_pct(m.operating_margin)}",
        f"- **Cash conversion (OCF/NI):** {fmt_ratio(m.cash_conversion, 'x')}",
        f"- **Owner earnings (est.):** {fmt_money(m.owner_earnings, m.currency)}",
        f"- **Owner earnings yield:** {fmt_pct(m.owner_earnings_yield)}",
        f"- **Free cash flow:** {fmt_money(m.fcf, m.currency)}",
        f"- **Debt / equity:** {fmt_ratio(m.debt_to_equity, '%' if (m.debt_to_equity or 0) > 5 else 'x')}",
        f"- **Interest coverage:** {fmt_ratio(m.interest_coverage, 'x')}",
        f"- **5Y ann. return / vol / Sharpe:** {fmt_pct(m.ann_return_5y)} / {fmt_pct(m.ann_vol_5y)} / {fmt_ratio(m.sharpe_5y)}",
        f"",
        f"## Valuation",
        f"- **Trailing P/E:** {fmt_ratio(m.trailing_pe, 'x')}",
        f"- **Forward P/E:** {fmt_ratio(m.forward_pe, 'x')}",
        f"- **Price / book:** {fmt_ratio(m.price_to_book, 'x')}",
        f"- **Dividend yield:** {fmt_ratio(m.dividend_yield, '%')}",
        f"- **Margin of safety:** {margin_of_safety_note(m)}",
        f"- **Recommended action:** Compare owner earnings yield to your required return; demand 30%+ discount for uncertain businesses.",
        f"",
        f"## Sell Criteria — Item-by-Item Check",
        f"1. **Price severely overvalued?** {'Yes' if (m.trailing_pe or 0) > 40 else 'No'} — trailing P/E {fmt_ratio(m.trailing_pe, 'x')}",
        f"2. **Fundamental moat destruction?** No — not detectable from snapshot; monitor ROIC trend",
        f"3. **Management integrity issue?** {'Yes — investigate' if not m.q_integrity else 'No — not flagged'}",
        f"4. **Significantly better opportunity available?** Unknown — compare opportunity cost manually",
        f"",
        f"## Key Risks (max 3)",
    ]

    risks = []
    if (m.debt_to_equity or 0) > 80:
        risks.append("Leverage — elevated debt/equity; stress-test −30% revenue scenario")
    if (m.ann_vol_5y or 0) > 0.25:
        risks.append("Volatility — 5Y annualised vol >25%; Mr. Market may offer better entry")
    if not m.q_moat:
        risks.append("Competitive position — ROIC/ROE below Buffett thresholds; moat may be weak")
    if not risks:
        risks = [
            "Concentration / macro — sector-specific downturn not modelled here",
            "Accounting — automated metrics cannot replace reading the annual report",
            "Valuation — paying fair price leaves little room for error",
        ]
    for i, r in enumerate(risks[:3], 1):
        lines.append(f"{i}. {r}")

    lines.extend(
        [
            f"",
            f"## Monitoring Indicators",
            f"- **Quarterly:** ROIC, cash conversion, debt/EBITDA, revenue trend",
            f"- **Annually:** Owner earnings vs market cap yield, moat narrative in shareholder letter",
            f"- **Sell trigger:** Integrity issue, ROIC < WACC for 3+ years, or price >2x estimated intrinsic value",
            f"",
            f"## Overall Assessment",
            f"{m.name} screens as a **{'quality' if m.q_moat else 'average'}** business at **{'attractive' if m.q_price else 'full or rich'}** prices.",
            f"Owner earnings yield of {fmt_pct(m.owner_earnings_yield)} and ROIC of {fmt_ratio(m.roic, '%')} are the numbers Buffett would study first.",
            f"Do not rely on this report alone — read the latest annual report and ask whether you would hold this for 10 years without checking the price.",
            f"",
        ]
    )

    if m.red_flags:
        lines.append("## ⚠ Red Flags")
        for f in m.red_flags:
            lines.append(f"- {f}")
        lines.append("")

    if m.warnings:
        lines.append("## Data Warnings")
        for w in m.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("---")
    lines.append("*Not investment advice. Verify all figures against official filings.*")
    return "\n".join(lines)


def main() -> int:
    global RISK_FREE_RATE
    parser = argparse.ArgumentParser(description="Buffett-style equity report from free Yahoo Finance data.")
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL, D05.SI, ES3.SI)")
    parser.add_argument("-o", "--output", help="Write report to file (default: stdout)")
    parser.add_argument("--rf", type=float, default=RISK_FREE_RATE, help="Risk-free rate decimal (default 0.021)")
    parser.add_argument("--json", action="store_true", help="Dump raw metrics JSON instead of report")
    args = parser.parse_args()

    RISK_FREE_RATE = args.rf

    if yf is None:
        print(
            "Error: yfinance is required.\n  Install: pip install yfinance\n",
            file=sys.stderr,
        )
        return 1

    try:
        ticker = normalize_ticker(args.ticker)
        metrics = fetch_with_yfinance(ticker)
        metrics = apply_quick_filter(metrics)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return 1

    if args.json:
        from dataclasses import asdict

        payload = asdict(metrics)
        text = json.dumps(payload, indent=2, default=str)
    else:
        text = generate_report(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
