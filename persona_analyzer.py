"""Persona + template analysis helpers — clustering, persona matching, stats, and HTML report."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Template
from openai import OpenAI
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Template detection
# ---------------------------------------------------------------------------

MIN_TEMPLATE_LENGTH = 80  # messages shorter than this are too short to cluster reliably


def normalize_message(text: str, recipient_name: str | None = None) -> str:
    """Replace recipient first name with [NAME] placeholder, then strip greeting.

    Handles multi-message outreach joined with ' ||| ' by merging into a single
    text block before normalizing.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace(" ||| ", "\n")
    if recipient_name and isinstance(recipient_name, str) and recipient_name.strip():
        text = re.sub(r'\b' + re.escape(recipient_name.strip()) + r'\b', '[NAME]', text)
    return strip_greeting(text)


def strip_greeting(text: str) -> str:
    """Remove personalized greeting line or inline greeting prefix.

    Handles both standalone greeting lines ('Hey Dan!\\n...') and inline
    greetings where the name is followed by a comma and more text
    ('Hey Dan, thanks for connecting...').
    """
    if not isinstance(text, str):
        return ""
    lines = text.strip().split("\n")
    if not lines:
        return ""
    first = lines[0].strip()
    greetings = r"(?:Hey|Hi|Hello|Dear|היי|הי|שלום|אהלן)"
    # Standalone greeting line (e.g. "Hey Dan!" or "היי איתי!")
    if re.match(rf"^{greetings}\s+\S+[,!.\s]*$", first, re.IGNORECASE):
        return "\n".join(lines[1:]).strip()
    # Inline greeting prefix (e.g. "Hey Dan, thanks!" or "היי איתי, נעים...")
    m = re.match(rf"^{greetings}\s+\S+[,!.]\s*", first, re.IGNORECASE)
    if m:
        remainder = first[m.end():]
        if remainder:
            lines[0] = remainder
            return "\n".join(lines).strip()
    return text.strip()


def cluster_templates(messages: list[str], threshold: int = 75,
                      recipient_names: list[str] | None = None) -> list[dict]:
    """Cluster message bodies into templates using fuzzy matching.

    Messages shorter than MIN_TEMPLATE_LENGTH after normalization are treated
    as one-off (not clustered) since fuzzy matching is unreliable on short strings.

    Returns list of dicts: {template_id, preview, message_indices}
    """
    if recipient_names is None:
        recipient_names = [None] * len(messages)
    stripped = [normalize_message(m, n) for m, n in zip(messages, recipient_names)]
    clusters: list[dict] = []

    for i, body in enumerate(stripped):
        if not body:
            continue
        if len(body) < MIN_TEMPLATE_LENGTH:
            clusters.append({
                "template_id": len(clusters) + 1,
                "preview": body,
                "message_indices": [i],
            })
            continue
        matched = False
        for cluster in clusters:
            rep = stripped[cluster["message_indices"][0]]
            if fuzz.ratio(body, rep) >= threshold:
                cluster["message_indices"].append(i)
                matched = True
                break
        if not matched:
            clusters.append({
                "template_id": len(clusters) + 1,
                "preview": body,
                "message_indices": [i],
            })

    return clusters


# ---------------------------------------------------------------------------
# Persona parsing
# ---------------------------------------------------------------------------

def _dedupe_titles(titles: list[str], threshold: int = 75) -> list[str]:
    """Fuzzy-deduplicate a list of titles, keeping one representative per cluster."""
    unique: list[str] = []
    for title in titles:
        if not title:
            continue
        if not any(fuzz.ratio(title.lower(), u.lower()) >= threshold for u in unique):
            unique.append(title)
    return unique


def generate_persona_prompt(results_df: pd.DataFrame) -> str:
    """Generate a prompt for AI to classify job titles into personas."""
    raw_titles = (
        results_df.drop_duplicates(subset=["Recipient URL"])["Position"]
        .dropna()
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )
    titles = _dedupe_titles(raw_titles, threshold=55)

    lines = [
        "Classify these job titles into persona groups (e.g. 'SE Leader', 'IC SE', 'VP Sales').",
        "Output format — one line per persona: persona_name,title_pattern1|title_pattern2|...",
        "",
        "Titles:",
    ]
    for title in titles:
        lines.append(f"  - {title}")
    lines.append("")
    lines.append("Output personas:")

    return "\n".join(lines)


def parse_persona_input(text: str) -> dict[str, list[str]]:
    """Parse AI persona output into {persona_name: [title_patterns]}."""
    personas = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",", 1)
        name = parts[0].strip()
        patterns = [p.strip() for p in parts[1].split("|") if p.strip()]
        if name and patterns:
            personas[name] = patterns
    return personas


def load_config(path: Path) -> dict | None:
    """Load config from a YAML file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or None


def classify_personas_via_api(results_df: pd.DataFrame, api_key: str) -> dict[str, list[str]] | None:
    """Call OpenAI API to classify job titles into personas."""
    prompt = generate_persona_prompt(results_df)
    title_count = prompt.count("\n  - ")
    print(f"  Sending {title_count} unique titles to OpenAI...")
    client = OpenAI(api_key=api_key, timeout=60)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You classify job titles into persona groups. "
                "Create 5-15 broad persona groups that cover all the titles. "
                "Output exactly one line per persona in the format: persona_name,title_pattern1|title_pattern2|..."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content
    if not text:
        return None
    return parse_persona_input(text) or None


def auto_cluster_personas(results_df: pd.DataFrame, threshold: int = 70) -> dict[str, list[str]]:
    """Auto-cluster job titles into persona groups using fuzzy matching."""
    raw_titles = (
        results_df.drop_duplicates(subset=["Recipient URL"])["Position"]
        .dropna()
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )

    clusters: list[list[str]] = []
    for title in raw_titles:
        matched = False
        for cluster in clusters:
            if fuzz.ratio(title.lower(), cluster[0].lower()) >= threshold:
                cluster.append(title)
                matched = True
                break
        if not matched:
            clusters.append([title])

    personas = {}
    for cluster in clusters:
        name = Counter(cluster).most_common(1)[0][0]
        patterns = list(dict.fromkeys(cluster))
        personas[name] = patterns

    return personas


def assign_persona(position: str, personas: dict[str, list[str]], threshold: int = 65) -> str:
    """Assign a persona to a position using fuzzy matching."""
    if not isinstance(position, str) or not position:
        return "Unknown"
    best_score = 0
    best_persona = "Unknown"
    for persona_name, patterns in personas.items():
        for pattern in patterns:
            score = fuzz.partial_ratio(position.lower(), pattern.lower())
            if score > best_score:
                best_score = score
                best_persona = persona_name
    return best_persona if best_score >= threshold else "Unknown"


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(results_df: pd.DataFrame, connections_df: pd.DataFrame,
                  templates: list[dict], first_messages: list[str],
                  personas: dict[str, list[str]]) -> dict:
    """Compute general, per-template, and per-persona statistics."""
    total_connections = len(connections_df)
    total_outreach = len(results_df)
    responded_df = results_df[results_df["Got Response"] == True]  # noqa: E712
    total_responses = len(responded_df)
    response_rate = (total_responses / total_outreach * 100) if total_outreach else 0
    response_rate_connections = (total_responses / total_connections * 100) if total_connections else 0
    avg_days = responded_df["Days to Response"].mean() if not responded_df.empty else 0

    # Date range
    dates = results_df["Outreach Date"].dropna()
    date_min = dates.min() if not dates.empty else None
    date_max = dates.max() if not dates.empty else None

    general = {
        "total_connections": total_connections,
        "total_outreach": total_outreach,
        "total_responses": total_responses,
        "response_rate": round(response_rate, 1),
        "response_rate_connections": round(response_rate_connections, 1),
        "avg_days": round(avg_days, 1),
        "date_range": (
            f"{date_min.strftime('%b %d, %Y') if date_min else '?'}"
            f" — "
            f"{date_max.strftime('%b %d, %Y') if date_max else '?'}"
        ),
    }

    # Build index -> got_response mapping
    got_response = {i: row["Got Response"] for i, row in results_df.iterrows()}

    # Per-template stats (only templates with at least 1 response)
    template_stats = []
    for cluster in templates:
        indices = cluster["message_indices"]
        sent = len(indices)
        responded = sum(1 for idx in indices if idx < len(results_df) and got_response.get(idx, False))
        if responded == 0:
            continue
        rate = (responded / sent * 100) if sent else 0
        template_stats.append({
            "id": cluster["template_id"],
            "preview": cluster["preview"],
            "sent": sent,
            "responded": responded,
            "rate": round(rate, 1),
        })

    # Assign personas to each row
    results_df = results_df.copy()
    results_df["Persona"] = results_df["Position"].apply(lambda p: assign_persona(p, personas))

    # Per-persona stats
    reported_ids = {t["id"] for t in template_stats}
    persona_stats = []
    for persona_name in sorted(set(results_df["Persona"])):
        pf = results_df[results_df["Persona"] == persona_name]
        p_responded = pf[pf["Got Response"] == True]  # noqa: E712
        contacted = len(pf)
        resp_count = len(p_responded)
        rate = (resp_count / contacted * 100) if contacted else 0
        avg_d = p_responded["Days to Response"].mean() if not p_responded.empty else 0
        companies = pf["Company"].dropna().unique()[:5].tolist()
        titles = pf["Position"].dropna().unique().tolist()

        # All templates used for this persona
        persona_templates = []
        for cluster in templates:
            indices = cluster["message_indices"]
            persona_indices = [
                idx for idx in indices
                if idx < len(results_df) and results_df.iloc[idx]["Persona"] == persona_name
            ]
            if persona_indices:
                t_sent = len(persona_indices)
                t_resp = sum(1 for idx in persona_indices if got_response.get(idx, False))
                persona_templates.append({
                    "id": cluster["template_id"],
                    "sent": t_sent,
                    "responded": t_resp,
                    "in_report": cluster["template_id"] in reported_ids,
                })
        persona_templates.sort(key=lambda t: t["responded"] / t["sent"] if t["sent"] else 0, reverse=True)

        persona_stats.append({
            "name": persona_name,
            "contacted": contacted,
            "responded": resp_count,
            "rate": round(rate, 1),
            "avg_days": round(avg_d, 1),
            "templates": persona_templates,
            "titles": ", ".join(titles) if titles else "—",
            "companies": ", ".join(companies) if companies else "—",
        })

    template_stats.sort(key=lambda t: t["rate"], reverse=True)
    persona_stats.sort(key=lambda p: p["rate"], reverse=True)

    # --- Insights ---

    # Day of week
    outreach_dates = pd.to_datetime(results_df["Outreach Date"])
    results_df["_day"] = outreach_dates.dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_stats = []
    for day in day_order:
        day_df = results_df[results_df["_day"] == day]
        if day_df.empty:
            continue
        d_sent = len(day_df)
        d_resp = int(day_df["Got Response"].sum())
        d_rate = round(d_resp / d_sent * 100, 1) if d_sent else 0
        day_stats.append({"day": day, "sent": d_sent, "responded": d_resp, "rate": d_rate})

    # Message length buckets
    content_col = "All Sent Content" if "All Sent Content" in results_df.columns else "Outreach Content"
    results_df["_msg_len"] = results_df[content_col].fillna("").str.len()
    length_bins = [(0, 100, "< 100"), (100, 300, "100-300"), (300, 600, "300-600"), (600, 99999, "600+")]
    length_stats = []
    for lo, hi, label in length_bins:
        bucket = results_df[(results_df["_msg_len"] >= lo) & (results_df["_msg_len"] < hi)]
        if bucket.empty:
            continue
        l_sent = len(bucket)
        l_resp = int(bucket["Got Response"].sum())
        l_rate = round(l_resp / l_sent * 100, 1) if l_sent else 0
        length_stats.append({"label": label, "sent": l_sent, "responded": l_resp, "rate": l_rate})

    # Response timing
    resp_days = responded_df["Days to Response"]
    timing_stats = {}
    if not resp_days.empty:
        timing_stats = {
            "median_hours": round(float(resp_days.median()) * 24, 1),
            "within_1h": int((resp_days <= 1 / 24).sum()),
            "within_1d": int((resp_days <= 1).sum()),
            "within_1w": int((resp_days <= 7).sum()),
            "total": len(resp_days),
        }

    return {
        "general": general,
        "templates": template_stats,
        "personas": persona_stats,
        "day_of_week": day_stats,
        "message_length": length_stats,
        "response_timing": timing_stats,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def render_report(stats: dict, output_path: str = "outreach_report.html"):
    """Render stats into an HTML report file."""
    template_path = Path(__file__).parent / "report_template.html"
    template = Template(template_path.read_text(encoding="utf-8"))
    html = template.render(stats=stats, now=datetime.now().strftime("%B %d, %Y"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
