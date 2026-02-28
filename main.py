"""OutreachAnalyzer — full pipeline: match outreach -> cluster templates -> persona analysis -> HTML report."""

__version__ = "1.0.0"

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from persona_analyzer import (
    auto_cluster_personas,
    classify_personas_via_api,
    cluster_templates,
    compute_stats,
    load_config,
    render_report,
)
from utils import detect_user_url, find_csv_files, load_connections_csv, load_messages_csv


def categorize_messages(messages_df: pd.DataFrame, user_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split messages into SENT (from user) and RECEIVED (to user)."""
    sent = messages_df[messages_df["SENDER PROFILE URL"] == user_url].copy()
    received = messages_df[messages_df["SENDER PROFILE URL"] != user_url].copy()
    return sent, received


def match_responses(messages_df: pd.DataFrame, user_url: str) -> list[dict]:
    """For each conversation, find the first response to user's outreach.

    Returns a list of dicts with outreach and response details.
    Skips conversations where the other person messaged first (not outreach).
    """
    results = []

    for conv_id, group in messages_df.groupby("CONVERSATION ID"):
        group = group.sort_values("DATE", ascending=True)

        first_msg = group.iloc[0]
        if first_msg["SENDER PROFILE URL"] != user_url:
            continue  # Other person initiated — not outreach

        response = None
        sent_before_response = []

        for _, row in group.iterrows():
            if row["SENDER PROFILE URL"] == user_url:
                if response is None:
                    sent_before_response.append(row)
            else:
                if response is None:
                    response = row

        if not sent_before_response:
            continue

        last_sent = sent_before_response[-1]
        first_sent = sent_before_response[0]

        recipient_name = first_sent.get("TO", "")
        recipient_url = first_sent.get("RECIPIENT PROFILE URLS", "")

        entry = {
            "Recipient Name": recipient_name,
            "Recipient URL": recipient_url,
            "Outreach Date": first_sent["DATE"],
            "Messages Sent": len(sent_before_response),
            "Outreach Content": first_sent.get("CONTENT", ""),
            "All Sent Content": " ||| ".join(
                str(m.get("CONTENT", "")) for m in sent_before_response
            ),
            "Conversation ID": conv_id,
        }

        if response is not None:
            entry["Response Date"] = response["DATE"]
            entry["Days to Response"] = (response["DATE"] - last_sent["DATE"]).total_seconds() / 86400
            entry["Response Content"] = response.get("CONTENT", "")
            entry["Got Response"] = True
        else:
            entry["Response Date"] = None
            entry["Days to Response"] = None
            entry["Response Content"] = ""
            entry["Got Response"] = False

        results.append(entry)

    return results


def enrich_with_connections(results: list[dict], connections_df: pd.DataFrame) -> pd.DataFrame:
    """Join outreach results with connection data (Company, Position)."""
    df = pd.DataFrame(results)
    if df.empty:
        return df

    conn = connections_df[["URL", "Company", "Position", "First Name", "Last Name"]].copy()
    conn["URL"] = conn["URL"].str.strip()
    df["Recipient URL"] = df["Recipient URL"].str.strip()

    merged = df.merge(conn, left_on="Recipient URL", right_on="URL", how="left")
    merged.drop(columns=["URL"], inplace=True, errors="ignore")

    col_order = [
        "Recipient Name", "First Name", "Last Name", "Company", "Position",
        "Recipient URL", "Outreach Date", "Response Date", "Days to Response",
        "Got Response", "Messages Sent", "Outreach Content", "All Sent Content",
        "Response Content", "Conversation ID",
    ]
    return merged[[c for c in col_order if c in merged.columns]]


def main():
    # --- Step 1: Load CSV data ---
    folder = input("Enter path to LinkedIn export folder: ").strip()
    if not folder:
        print("No folder provided. Exiting.")
        sys.exit(1)

    try:
        csv_files = find_csv_files(folder)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Found connections: {csv_files['connections']}")
    print(f"Found messages: {csv_files['messages']}")

    connections_df = load_connections_csv(csv_files["connections"])
    messages_df = load_messages_csv(csv_files["messages"])

    print(f"Loaded {len(connections_df)} connections, {len(messages_df)} messages")

    # --- Step 2: Date filter (default: 6 months ago) ---
    default_since = (datetime.now() - relativedelta(months=6)).strftime("%d/%m/%Y")
    since_str = input(f"Analyze messages since date (DD/MM/YYYY, default {default_since}): ").strip()
    if not since_str:
        since_str = default_since
    try:
        since_date = pd.Timestamp(datetime.strptime(since_str, "%d/%m/%Y"), tz="UTC")
        messages_df = messages_df[messages_df["DATE"] >= since_date]
        print(f"Filtering messages from {since_date.strftime('%b %d, %Y')} onwards -> {len(messages_df)} messages")
    except ValueError:
        print("Invalid date format, using all messages.")

    user_url = detect_user_url(messages_df)
    print(f"Detected user: {user_url}")

    # --- Step 3: Match outreach to responses ---
    results = match_responses(messages_df, user_url)
    print(f"\nFound {len(results)} outreach conversations")

    responded = sum(1 for r in results if r["Got Response"])
    if results:
        print(f"  {responded} got responses ({responded / len(results) * 100:.1f}% response rate)")

    results_df = enrich_with_connections(results, connections_df)

    # Save Excel
    excel_df = results_df.copy()
    for col in excel_df.select_dtypes(include=["datetimetz"]).columns:
        excel_df[col] = excel_df[col].dt.tz_localize(None)
    excel_path = "outreach_results.xlsx"
    excel_df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"Results saved to {excel_path}")

    # --- Step 4: Load config ---
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)

    # --- Step 5: Template clustering ---
    content_col = "All Sent Content" if "All Sent Content" in results_df.columns else "Outreach Content"
    outreach_messages = results_df[content_col].fillna("").tolist()
    recipient_names = results_df["First Name"].fillna("").tolist() if "First Name" in results_df.columns else None
    min_msgs = (config or {}).get("min_template_messages", 2)
    templates = cluster_templates(outreach_messages, recipient_names=recipient_names)
    total_templates = len(templates)
    templates = [t for t in templates if len(t["message_indices"]) >= min_msgs]
    print(f"\nDetected {total_templates} message templates, {len(templates)} with >= {min_msgs} messages")

    # --- Step 6: Persona classification ---
    api_key = config.get("openai_api_key") if config else None
    personas = None

    if api_key:
        print("\nClassifying personas via OpenAI API...")
        try:
            personas = classify_personas_via_api(results_df, api_key)
            if personas:
                print(f"Classified {len(personas)} personas: {', '.join(personas.keys())}")
        except Exception as e:
            print(f"API call failed: {e}")

    if not personas:
        print("\nAuto-clustering personas from job titles (add openai_api_key to config.yaml for AI classification)")
        personas = auto_cluster_personas(results_df)
        print(f"Clustered {len(personas)} persona groups")

    # --- Step 7: Compute stats & render report ---
    stats = compute_stats(results_df, connections_df, templates, outreach_messages, personas)
    min_contacts = (config or {}).get("min_persona_contacts", 2)
    stats["personas"] = [
        p for p in stats["personas"]
        if p["contacted"] >= min_contacts and p["templates"]
    ]
    output = render_report(stats)
    print(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()
