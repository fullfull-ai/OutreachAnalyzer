"""End-to-end and unit tests for the OutreachAnalyzer pipeline."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import categorize_messages, enrich_with_connections, match_responses
from persona_analyzer import (
    _dedupe_titles,
    assign_persona,
    auto_cluster_personas,
    cluster_templates,
    compute_stats,
    load_config,
    normalize_message,
    parse_persona_input,
    render_report,
    strip_greeting,
)
from utils import detect_user_url, find_csv_files, load_connections_csv, load_messages_csv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).parent
USER_URL = "https://www.linkedin.com/in/matan-dobr"


@pytest.fixture
def connections_df():
    return load_connections_csv(str(TEST_DIR / "Connections.csv"))


@pytest.fixture
def messages_df():
    return load_messages_csv(str(TEST_DIR / "messages.csv"))


@pytest.fixture
def results_df(messages_df, connections_df):
    user_url = detect_user_url(messages_df)
    results = match_responses(messages_df, user_url)
    return enrich_with_connections(results, connections_df)


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_find_csv_files(self):
        files = find_csv_files(str(TEST_DIR))
        assert "connections" in files
        assert "messages" in files

    def test_find_csv_files_missing_folder(self):
        with pytest.raises(FileNotFoundError):
            find_csv_files("/nonexistent/path")

    def test_load_connections_csv(self, connections_df):
        assert len(connections_df) > 0
        assert "First Name" in connections_df.columns
        assert "Position" in connections_df.columns

    def test_load_messages_csv(self, messages_df):
        assert len(messages_df) > 0
        assert "CONVERSATION ID" in messages_df.columns
        assert pd.api.types.is_datetime64_any_dtype(messages_df["DATE"])

    def test_detect_user_url(self, messages_df):
        url = detect_user_url(messages_df)
        assert url == USER_URL


# ---------------------------------------------------------------------------
# Outreach matching tests
# ---------------------------------------------------------------------------

class TestMatchResponses:
    def test_finds_outreach_conversations(self, messages_df):
        results = match_responses(messages_df, USER_URL)
        assert len(results) > 0

    def test_response_count(self, messages_df):
        results = match_responses(messages_df, USER_URL)
        responded = sum(1 for r in results if r["Got Response"])
        assert responded > 0
        assert responded <= len(results)

    def test_skips_inbound_conversations(self, messages_df):
        results = match_responses(messages_df, USER_URL)
        # All results should be outreach (user sent first)
        for r in results:
            assert r["Outreach Content"] or r["All Sent Content"]

    def test_multi_message_outreach_merged(self, messages_df):
        results = match_responses(messages_df, USER_URL)
        multi = [r for r in results if r["Messages Sent"] > 1]
        for r in multi:
            assert " ||| " in r["All Sent Content"]

    def test_days_to_response_positive(self, messages_df):
        results = match_responses(messages_df, USER_URL)
        for r in results:
            if r["Got Response"]:
                assert r["Days to Response"] is not None

    def test_enrich_adds_position(self, results_df):
        assert "Position" in results_df.columns
        assert "Company" in results_df.columns


# ---------------------------------------------------------------------------
# Template clustering tests
# ---------------------------------------------------------------------------

class TestTemplateClustering:
    def test_cluster_templates_basic(self):
        long_a = "I'm reaching out because I noticed your work in cloud infrastructure and would love to discuss potential collaboration opportunities with our team"
        long_b = "We are launching a brand new product line for enterprise customers and looking for beta testers who match your profile exactly"
        msgs = [long_a, long_a, long_b]
        clusters = cluster_templates(msgs)
        assert len(clusters) == 2
        assert len(clusters[0]["message_indices"]) == 2

    def test_no_truncation(self):
        long_msg = "A" * 500
        clusters = cluster_templates([long_msg])
        assert clusters[0]["preview"] == long_msg

    def test_multi_message_separator_handled(self):
        msg = "Hey Dan ||| Here is my pitch about our product"
        normalized = normalize_message(msg, "Dan")
        assert " ||| " not in normalized
        assert "pitch" in normalized.lower()

    def test_strip_greeting_standalone(self):
        assert strip_greeting("Hey Dan!\nActual content") == "Actual content"
        assert strip_greeting("Hi John,\nActual content") == "Actual content"

    def test_strip_greeting_inline(self):
        result = strip_greeting("Hey Dan, thanks for connecting!")
        assert result == "thanks for connecting!"

    def test_strip_greeting_no_greeting(self):
        assert strip_greeting("Just a normal message") == "Just a normal message"

    def test_normalize_replaces_name(self):
        result = normalize_message("Hey Dan, great to meet you Dan", "Dan")
        assert "Dan" not in result
        assert "[NAME]" in result

    def test_with_test_data(self, results_df):
        content_col = "All Sent Content" if "All Sent Content" in results_df.columns else "Outreach Content"
        messages = results_df[content_col].fillna("").tolist()
        names = results_df["First Name"].fillna("").tolist() if "First Name" in results_df.columns else None
        clusters = cluster_templates(messages, recipient_names=names)
        assert len(clusters) >= 2
        # Main template should have the most messages
        biggest = max(clusters, key=lambda c: len(c["message_indices"]))
        assert len(biggest["message_indices"]) >= 20


# ---------------------------------------------------------------------------
# Persona tests
# ---------------------------------------------------------------------------

class TestPersonas:
    def test_dedupe_titles(self):
        titles = ["Sales Engineer", "Senior Sales Engineer", "Solutions Architect", "Sales Engineer"]
        deduped = _dedupe_titles(titles)
        assert len(deduped) < len(titles)

    def test_parse_persona_input(self):
        text = "SE,Sales Engineer|Solutions Engineer\nLeader,VP|Director"
        personas = parse_persona_input(text)
        assert len(personas) == 2
        assert "SE" in personas
        assert "Sales Engineer" in personas["SE"]

    def test_assign_persona(self):
        personas = {"SE": ["Sales Engineer", "Solutions Engineer"], "Leader": ["VP", "Director"]}
        assert assign_persona("Senior Sales Engineer", personas) == "SE"
        assert assign_persona("VP of Sales", personas) == "Leader"
        assert assign_persona("", personas) == "Unknown"

    def test_auto_cluster(self, results_df):
        personas = auto_cluster_personas(results_df)
        assert len(personas) >= 2
        for name, patterns in personas.items():
            assert len(patterns) >= 1

    def test_load_config_missing_file(self):
        result = load_config(Path("/nonexistent/config.yaml"))
        assert result is None

    def test_load_config_valid(self, tmp_path):
        import yaml
        config_file = tmp_path / "config.yaml"
        config_file.write_text("openai_api_key: sk-test123\n")
        result = load_config(config_file)
        assert result is not None
        assert result["openai_api_key"] == "sk-test123"

    def test_load_config_empty_key(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("openai_api_key:\n")
        result = load_config(config_file)
        # yaml parses empty value as None, so config dict exists but key is None
        assert result is None or result.get("openai_api_key") is None


# ---------------------------------------------------------------------------
# Stats & report tests
# ---------------------------------------------------------------------------

class TestStatsAndReport:
    def test_compute_stats(self, results_df, connections_df):
        messages = results_df["All Sent Content"].fillna("").tolist()
        names = results_df["First Name"].fillna("").tolist()
        templates = cluster_templates(messages, recipient_names=names)
        personas = auto_cluster_personas(results_df)
        stats = compute_stats(results_df, connections_df, templates, messages, personas)

        assert "general" in stats
        assert "templates" in stats
        assert "personas" in stats
        assert stats["general"]["total_outreach"] == len(results_df)
        assert stats["general"]["response_rate"] == round(
            sum(results_df["Got Response"]) / len(results_df) * 100, 1
        )

    def test_templates_sorted_by_rate(self, results_df, connections_df):
        messages = results_df["All Sent Content"].fillna("").tolist()
        names = results_df["First Name"].fillna("").tolist()
        templates = cluster_templates(messages, recipient_names=names)
        personas = auto_cluster_personas(results_df)
        stats = compute_stats(results_df, connections_df, templates, messages, personas)

        rates = [t["rate"] for t in stats["templates"]]
        assert rates == sorted(rates, reverse=True)

    def test_personas_sorted_by_rate(self, results_df, connections_df):
        messages = results_df["All Sent Content"].fillna("").tolist()
        names = results_df["First Name"].fillna("").tolist()
        templates = cluster_templates(messages, recipient_names=names)
        personas = auto_cluster_personas(results_df)
        stats = compute_stats(results_df, connections_df, templates, messages, personas)

        rates = [p["rate"] for p in stats["personas"]]
        assert rates == sorted(rates, reverse=True)

    def test_render_report(self, results_df, connections_df, tmp_path):
        messages = results_df["All Sent Content"].fillna("").tolist()
        names = results_df["First Name"].fillna("").tolist()
        templates = cluster_templates(messages, recipient_names=names)
        personas = auto_cluster_personas(results_df)
        stats = compute_stats(results_df, connections_df, templates, messages, personas)

        output = str(tmp_path / "test_report.html")
        result = render_report(stats, output_path=output)
        assert os.path.exists(result)

        html = Path(result).read_text()
        assert "Salora AI" in html
        assert "salora.ai" in html
        assert "<table>" in html
