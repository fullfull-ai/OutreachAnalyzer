# OutreachAnalyzer by Salora AI

**v1.0.0**

Analyze your LinkedIn outreach effectiveness. Takes your LinkedIn export CSVs (Connections + Messages), matches outreach to responses, clusters messages into templates, and generates an HTML report with per-persona and per-template stats.

## Quick Start

```bash
pip install -r requirements.txt

python3 main.py
# Enter your LinkedIn export folder path when prompted (e.g. "tests")
# Enter a start date (DD/MM/YYYY) or press Enter for default (6 months ago)
# -> outputs outreach_results.xlsx + outreach_report.html
```

## LinkedIn Export Setup

Export your data from LinkedIn (Settings > Data privacy > Get a copy of your data) and place these two files in a folder:

| File | Key Columns |
|------|-------------|
| `Connections.csv` | First Name, Last Name, URL, Company, Position |
| `messages.csv` | Conversation ID, From, Sender Profile URL, Date, Content |

## Persona Classification

Personas are classified automatically using fuzzy title clustering. For better AI-powered classification, add your OpenAI API key to `config.yaml`:

```yaml
openai_api_key: sk-...
```

## Requirements

- Python 3.11+
- pandas, openpyxl, rapidfuzz, jinja2, pyyaml
- openai (optional, for AI persona classification)

## Disclaimer

This project is a **proof of concept** released as an open-source research tool by [Salora AI](https://www.salora.ai). It is provided "as is", without warranty of any kind, express or implied. Salora AI and the contributors assume no responsibility or liability for any errors, issues, or damages arising from the use of this tool. Use it at your own risk.

## License

This project is licensed under the [MIT License](LICENSE).

---

Built by [Salora AI](https://www.salora.ai)
