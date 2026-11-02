"""Shared presentation helpers: formatting, styling and cards."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence

import streamlit as st

ACCENT = "#E50914"
GOLD = "#F5C518"
MUTED = "#9AA0A6"
SURFACE = "#16161D"
PAGE = "#0B0B0F"

CSS = f"""
<style>
  .block-container {{ padding-top: 2.0rem; padding-bottom: 3.5rem; max-width: 1320px; }}
  [data-testid="stToolbar"] {{ display: none; }}
  #MainMenu {{ visibility: hidden; }}
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #141419 0%, #0B0B0F 100%);
      border-right: 1px solid rgba(255,255,255,0.06);
      width: 260px !important;
      min-width: 260px !important;
  }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  .cs-hero {{
      padding: 1.4rem 1.6rem 1.5rem 1.6rem;
      border-radius: 18px;
      background:
        radial-gradient(1100px 380px at 0% 0%, rgba(229,9,20,0.22), transparent 60%),
        linear-gradient(135deg, #17171F 0%, #101014 100%);
      border: 1px solid rgba(255,255,255,0.07);
      margin-bottom: 1.1rem;
  }}
  .cs-kicker {{ color: {ACCENT}; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
  .cs-hero h1 {{ margin: .25rem 0 .35rem 0; font-size: 2.15rem; }}
  .cs-hero p {{ color: {MUTED}; margin: 0; max-width: 60rem; }}
  .cs-section {{ margin: 1.3rem 0 .35rem 0; }}
  .cs-section h2 {{ font-size: 1.25rem; margin: 0; }}
  .cs-section p {{ color: {MUTED}; margin: .15rem 0 0 0; font-size: .88rem; }}
  .cs-card {{
      border-radius: 14px; overflow: hidden; height: 100%; display: flex; flex-direction: column;
      background: linear-gradient(180deg, #191920 0%, #131318 100%);
      border: 1px solid rgba(255,255,255,0.07);
  }}
  .cs-grid {{ display: grid; gap: .9rem; margin: .4rem 0 .6rem 0; }}
  .cs-poster {{
      position: relative; height: 230px; background-size: cover; background-position: center;
      background-color: #1E1E26;
  }}
  .cs-poster-fallback {{
      display: flex; align-items: center; justify-content: center; text-align: center;
      font-weight: 800; font-size: 2.1rem; letter-spacing: .08em;
      background: linear-gradient(150deg, #2A2A36 0%, #15151C 100%);
      color: rgba(255,255,255,0.55);
  }}
  .cs-match {{
      position: absolute; top: .55rem; left: .55rem;
      background: rgba(0,0,0,0.72); color: #fff; font-size: .72rem; font-weight: 700;
      padding: .18rem .45rem; border-radius: 6px; backdrop-filter: blur(4px);
  }}
  .cs-body {{ padding: .7rem .8rem .85rem .8rem; }}
  .cs-title {{ font-weight: 700; font-size: .95rem; line-height: 1.25; margin-bottom: .2rem; }}
  .cs-meta {{ color: {MUTED}; font-size: .76rem; }}
  .cs-reasons {{ margin-top: .45rem; display: flex; flex-wrap: wrap; gap: .25rem; }}
  .cs-chip {{
      font-size: .68rem; color: #C9CBD3; background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.06); padding: .1rem .4rem; border-radius: 999px;
  }}
  .cs-big-number {{ font-size: 2.5rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.05; }}
  .cs-subtle {{ color: {MUTED}; font-size: .85rem; }}
  .cs-badge {{
      display: inline-block; font-size: .74rem; padding: .18rem .5rem; border-radius: 999px;
      background: rgba(229,9,20,.14); color: #FF6B72; border: 1px solid rgba(229,9,20,.3);
  }}
  .cs-footer {{ color: #6B6F76; font-size: .74rem; margin-top: 2rem; }}
  div[data-testid="stMetric"] {{
      background: linear-gradient(180deg, #17171E 0%, #121217 100%);
      border: 1px solid rgba(255,255,255,0.07);
      padding: .7rem .8rem; border-radius: 12px;
  }}
  div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .76rem; line-height: 1.2; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.45rem; white-space: nowrap; }}
  div[data-testid="stMetricDelta"] {{ font-size: .72rem; }}
  .stButton > button {{ border-radius: 10px; font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def money(value: float | int | None, decimals: int = 1) -> str:
    """Compact currency: $1.2B, $84.3M, $950K."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if value != value:  # NaN
        return "—"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:.{decimals}f}B"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.{decimals}f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.0f}K"
    return f"{sign}${value:,.0f}"


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="cs-hero">
              <div class="cs-kicker">{html.escape(kicker)}</div>
              <h1>{html.escape(title)}</h1>
              <p>{html.escape(subtitle)}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="cs-section"><h2>{html.escape(title)}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def chips(values: Iterable[str], limit: int = 4) -> str:
    items = [v for v in values if v][:limit]
    return "".join(f'<span class="cs-chip">{html.escape(str(v))}</span>' for v in items)


def poster_card(
    *,
    title: str,
    meta: str,
    poster_url: str | None = None,
    badge: str | None = None,
    reasons: Sequence[str] = (),
) -> str:
    """Render-safe HTML for a single movie card."""
    safe_title = html.escape(title)
    if poster_url:
        poster = (
            f'<div class="cs-poster" style="background-image:url(\'{html.escape(poster_url)}\')">'
        )
    else:
        initials = "".join(word[0] for word in title.split()[:2]).upper() or "🎬"
        poster = f'<div class="cs-poster cs-poster-fallback"><span>{html.escape(initials)}</span>'
    badge_html = f'<span class="cs-match">{html.escape(badge)}</span>' if badge else ""
    return (
        f'<div class="cs-card">{poster}{badge_html}</div>'
        f'<div class="cs-body"><div class="cs-title">{safe_title}</div>'
        f'<div class="cs-meta">{html.escape(meta)}</div>'
        f'<div class="cs-reasons">{chips(reasons)}</div></div></div>'
    )


def card_grid(cards: Sequence[str], min_width: int = 185) -> None:
    """Lay out pre-rendered card HTML in a responsive CSS grid."""
    if not cards:
        return
    st.markdown(
        f'<div class="cs-grid" style="grid-template-columns:repeat(auto-fill,minmax({min_width}px,1fr))">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def empty_state(message: str) -> None:
    st.info(message, icon="🎬")
