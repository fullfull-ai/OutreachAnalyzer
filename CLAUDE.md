# OutreachAnalyzer

LinkedIn outreach effectiveness analyzer. Takes two LinkedIn export CSVs (Connections + Messages), matches outreach to responses, clusters messages into templates, and generates an HTML report with per-persona and per-template stats. Branded as "OutreachAnalyzer by Salora AI".

## Project Structure

```
OutreachAnalyzer/
  main.py              # Full pipeline: outreach matching → templates → personas → report
  persona_analyzer.py  # Helper module: clustering, persona matching, stats, HTML rendering
  utils.py             # Shared CSV loading, user detection helpers
  report_template.html # Jinja2 HTML template for the report
  requirements.txt     # pandas, openpyxl, rapidfuzz, jinja2
  tests/
    Connections.csv    # Test data (6450 connections)
    messages.csv       # Test data (299 messages)
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python3 main.py
# → prompts for LinkedIn export folder path (e.g. "tests")
# → optional date filter
# → matches outreach to responses → outputs outreach_results.xlsx
# → clusters message templates
# → prints a prompt to paste into an LLM for persona classification
# → paste the LLM response back (format: persona_name,pattern1|pattern2), then blank line
# → outputs outreach_report.html
```

## Python Version

Use python3.11 (`/opt/homebrew/bin/python3.11`) — has all dependencies installed.

## Key Design Decisions

- **User detection**: Most frequent URL in first 20 rows of messages CSV
- **Outreach = user sent first**: Conversations where someone else messaged first are skipped
- **Response matching**: First non-user message in a conversation after user's outreach
- **Template clustering**: Uses all sent messages (multi-message outreach merged), strips greeting line, then fuzzy match with `rapidfuzz.fuzz.ratio` threshold 75
- **Persona assignment**: `fuzz.partial_ratio` on position vs title patterns, threshold 65
- **LinkedIn CSV format**: Connections CSV has 3 metadata rows before header; messages have UTC timestamps

## CSV Formats

**Connections.csv** (3 skip rows, then):
`First Name, Last Name, URL, Email Address, Company, Position, Connected On`

**messages.csv**:
`CONVERSATION ID, CONVERSATION TITLE, FROM, SENDER PROFILE URL, TO, RECIPIENT PROFILE URLS, DATE, SUBJECT, CONTENT, FOLDER, ATTACHMENTS`

## Test Verification

- Dan Suleiman conversation (`2-NGU4ZjQ0MGIt...`) should show response matched to Matan's outreach
- Expected: 75 outreach conversations, 12 responses (16.0% rate) with test data
- Main template (#1) clusters ~56 messages, variant (#2) captures ~9
