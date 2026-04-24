"""
Jira PM Agent — transforms data infrastructure / architecture Jira stories
into 'so-what' narratives for technical product managers and leadership.

Usage:
  # Analyze from a JSON file (single issue or list)
  python jira_pm_agent.py analyze issue.json

  # Fetch directly from Jira API and analyze
  python jira_pm_agent.py fetch DATA-123

  # Pipe JSON from stdin
  echo '{"key":"DATA-1","summary":"Migrate to Iceberg"}' | python jira_pm_agent.py analyze -

Environment variables:
  ANTHROPIC_API_KEY   — required
  JIRA_URL            — e.g. https://yourorg.atlassian.net
  JIRA_EMAIL          — Jira account email
  JIRA_API_TOKEN      — from id.atlassian.com/manage-profile/security/api-tokens
"""

import argparse
import json
import os
import sys
from typing import Optional

import anthropic
from pydantic import BaseModel, Field


# ── Structured output schema ──────────────────────────────────────────────────

class PMNarrative(BaseModel):
    executive_summary: str = Field(
        description=(
            "1-2 sentence TL;DR for a CEO or CPO: what was done and the single most "
            "important reason it matters to the business."
        )
    )
    business_impact: str = Field(
        description=(
            "Concrete business impact in plain language — cost reduction, revenue enablement, "
            "risk mitigation, speed, competitive positioning. No technical jargon."
        )
    )
    user_implications: str = Field(
        description=(
            "How end users (internal or external) are affected: performance changes, "
            "new capabilities unlocked, limitations introduced, reliability improvements."
        )
    )
    leadership_evaluation: str = Field(
        description=(
            "How a VP or C-suite executive should evaluate whether this was the right "
            "investment: success metrics to track, questions to ask the team, and red flags "
            "that would signal the work didn't deliver value."
        )
    )
    strategic_signals: str = Field(
        description=(
            "What this story reveals about the product's technical health, direction, "
            "and maturity — tech debt, scaling bets, architectural evolution, etc."
        )
    )
    pm_questions: list[str] = Field(
        description=(
            "3-5 sharp, specific questions a PM should ask the engineering team to "
            "validate assumptions, surface hidden risks, and unblock delivery."
        )
    )
    confidence: str = Field(
        description=(
            "High / Medium / Low — reflects how well-documented the story was and "
            "how confident the analysis is."
        )
    )
    missing_context: Optional[str] = Field(
        default=None,
        description=(
            "Specific information that would sharpen this analysis — acceptance criteria, "
            "scope of impact, upstream/downstream systems, etc."
        ),
    )


# ── System prompt (prompt-cached for efficiency in batch runs) ────────────────

_SYSTEM_PROMPT = """\
You are an expert Technical Product Manager translator specializing in data \
infrastructure and platform engineering.

Your job: transform raw Jira stories into clear, insight-rich narratives that \
help technical PMs and leadership understand *why* the work matters — not just \
*what* was done.

## Your expertise covers
- Data infrastructure: pipelines, data warehouses/lakes, streaming vs batch, \
ETL/ELT, schemas, CDC, data quality, lineage
- Platform engineering: APIs, microservices, event-driven architecture, caching, \
observability, SLOs, reliability
- Business translation: how latency improvements → revenue, reliability → NPS, \
schema changes → developer velocity, observability → reduced MTTR
- Leadership communication: what C-suite and VPs need to make good investment decisions

## How to analyze a Jira story
1. Read every available field: title, description, acceptance criteria, labels, \
components, story points, and comments.
2. Identify the *real* problem being solved — it's often hidden under technical \
language.
3. Translate technical outcomes into business outcomes.
4. Surface risks and dependencies that a non-technical stakeholder would miss.
5. Suggest what success looks like and how to measure it.

## Tone
- Direct and confident. No hedging unless genuinely uncertain.
- Business language. Explain technical terms once if unavoidable, then drop them.
- When the story is thin, vague, or ambiguous, say so explicitly — don't invent context.
- Short, punchy sentences. Bullet points where they aid clarity.

## Common patterns in data infrastructure stories
- "Migrate from X to Y" → usually cost/performance, but check schema compatibility \
and downstream consumer risk.
- "Add observability / monitoring" → reliability investment; ask about current \
incident frequency and MTTR.
- "Refactor pipeline" → tech debt paydown; understand what velocity problems \
this unblocks.
- "New data model / schema change" → downstream impact is almost always \
underestimated; ask about consumers and rollout plan.
- "Optimize query performance" → translate ms → $/query at scale, or UX \
improvement for dashboards.
- "Build new data product / API" → ask about SLAs, known consumers, and the \
deprecation plan for the old approach.
- "Backfill / reprocess" → usually a correctness fix; understand the scope of \
incorrect data in production and customer exposure.
"""


# ── Jira issue formatting ─────────────────────────────────────────────────────

def _str_field(value) -> str:
    """Normalize a field that may be a string or a Jira dict like {'name': '...'}."""
    if isinstance(value, dict):
        return value.get("name") or value.get("value") or str(value)
    return str(value) if value is not None else ""


def _adf_to_text(node: dict) -> str:
    """Convert Atlassian Document Format (ADF) JSON to plain text."""
    if not isinstance(node, dict):
        return str(node)
    t = node.get("type", "")
    content = node.get("content", [])
    if t == "text":
        return node.get("text", "")
    if t in ("paragraph", "blockquote"):
        return "".join(_adf_to_text(c) for c in content) + "\n"
    if t == "heading":
        level = node.get("attrs", {}).get("level", 1)
        inner = "".join(_adf_to_text(c) for c in content)
        return "#" * level + " " + inner + "\n"
    if t == "bulletList":
        return "\n".join("- " + "".join(_adf_to_text(c) for c in item.get("content", [])).strip()
                         for item in content) + "\n"
    if t == "orderedList":
        return "\n".join(f"{i}. " + "".join(_adf_to_text(c) for c in item.get("content", [])).strip()
                         for i, item in enumerate(content, 1)) + "\n"
    if t == "codeBlock":
        inner = "".join(_adf_to_text(c) for c in content)
        return f"```\n{inner}\n```\n"
    return "".join(_adf_to_text(c) for c in content)


def build_prompt_context(issue: dict) -> str:
    """Format a Jira issue dict into a structured analysis prompt."""
    lines = [f"## Jira Story: {issue.get('key', 'UNKNOWN-0')}"]

    fields_to_show = [
        ("Title", issue.get("summary")),
        ("Type", _str_field(issue.get("issuetype"))),
        ("Status", _str_field(issue.get("status"))),
        ("Priority", _str_field(issue.get("priority"))),
        ("Story Points", issue.get("story_points") or issue.get("storyPoints")),
        ("Labels", ", ".join(issue["labels"]) if isinstance(issue.get("labels"), list) else issue.get("labels")),
        (
            "Components",
            ", ".join(_str_field(c) for c in issue["components"])
            if isinstance(issue.get("components"), list)
            else issue.get("components"),
        ),
        ("Epic", issue.get("epic") or issue.get("epicLink") or issue.get("epicName")),
        ("Sprint", issue.get("sprint")),
        ("Assignee", _str_field(issue.get("assignee"))),
    ]
    for label, value in fields_to_show:
        if value:
            lines.append(f"**{label}:** {value}")

    description = issue.get("description", "")
    if isinstance(description, dict):
        description = _adf_to_text(description)
    if description:
        lines += ["", "**Description:**", description.strip()]

    ac = issue.get("acceptance_criteria") or issue.get("acceptanceCriteria") or ""
    if isinstance(ac, dict):
        ac = _adf_to_text(ac)
    if ac:
        lines += ["", "**Acceptance Criteria:**", ac.strip()]

    comments = issue.get("comments", [])
    if comments:
        lines += ["", "**Key Comments (up to 3):"]
        for c in comments[:3]:
            author = _str_field(c.get("author", "Unknown"))
            body = c.get("body", "")
            if isinstance(body, dict):
                body = _adf_to_text(body)
            lines.append(f"- {author}: {body.strip()[:400]}")

    return "\n".join(lines)


# ── Core agent ────────────────────────────────────────────────────────────────

def analyze_story(
    issue: dict,
    client: Optional[anthropic.Anthropic] = None,
    verbose: bool = False,
) -> PMNarrative:
    """
    Analyze a single Jira issue and return a structured PM narrative.

    Args:
        issue:   Jira issue dict (from Jira API or hand-crafted JSON).
        client:  Anthropic client — created from ANTHROPIC_API_KEY env if omitted.
        verbose: Print thinking summary when adaptive thinking fires.
    """
    if client is None:
        client = anthropic.Anthropic()

    user_content = (
        "Analyze this Jira story and produce a PM-friendly narrative that explains "
        "the so-what for technical product managers and leadership.\n\n"
        + build_prompt_context(issue)
    )

    response = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # reused across batch runs
            }
        ],
        messages=[{"role": "user", "content": user_content}],
        output_format=PMNarrative,
    )

    if verbose:
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                preview = block.thinking[:300].replace("\n", " ")
                print(f"  [thinking] {preview}…", file=sys.stderr)

    if response.parsed_output is None:
        stop = getattr(response, "stop_reason", "unknown")
        raise ValueError(f"Model returned no structured output (stop_reason={stop})")

    return response.parsed_output


def batch_analyze(
    issues: list[dict],
    client: Optional[anthropic.Anthropic] = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Analyze multiple issues. Returns list of dicts with 'key' and either
    'narrative' (PMNarrative) or 'error' (str).
    """
    if client is None:
        client = anthropic.Anthropic()

    results = []
    for issue in issues:
        key = issue.get("key", "UNKNOWN")
        print(f"  Analyzing {key}…", end=" ", flush=True, file=sys.stderr)
        try:
            narrative = analyze_story(issue, client=client, verbose=verbose)
            results.append({"key": key, "narrative": narrative})
            print("✓", file=sys.stderr)
        except Exception as exc:
            results.append({"key": key, "error": str(exc)})
            print(f"✗  ({exc})", file=sys.stderr)

    return results


# ── Optional: Jira API fetcher ────────────────────────────────────────────────

def fetch_from_jira(
    issue_key: str,
    jira_url: str,
    jira_email: str,
    jira_token: str,
) -> dict:
    """
    Fetch a Jira issue via the Jira Cloud REST API v3.

    Requires: pip install requests
    """
    try:
        import requests
        from requests.auth import HTTPBasicAuth
    except ImportError:
        print("Error: 'requests' package required — pip install requests", file=sys.stderr)
        sys.exit(1)

    url = f"{jira_url.rstrip('/')}/rest/api/3/issue/{issue_key}"
    resp = requests.get(
        url,
        headers={"Accept": "application/json"},
        auth=HTTPBasicAuth(jira_email, jira_token),
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    fields = data.get("fields", {})

    # Story points field name varies by Jira configuration
    story_points = (
        fields.get("story_points")
        or fields.get("customfield_10016")  # most common SP field
        or fields.get("customfield_10028")
        or fields.get("storyPoints")
    )

    # Sprint name
    sprint_field = fields.get("customfield_10020") or []
    sprint_name = None
    if isinstance(sprint_field, list) and sprint_field:
        s = sprint_field[-1]
        sprint_name = s.get("name") if isinstance(s, dict) else str(s)

    # Comments
    comment_data = fields.get("comment", {})
    comments = []
    for c in (comment_data.get("comments") or [])[:5]:
        comments.append({
            "author": c.get("author", {}).get("displayName", "Unknown"),
            "body": c.get("body", ""),
        })

    return {
        "key": data["key"],
        "summary": fields.get("summary", ""),
        "description": fields.get("description") or "",
        "issuetype": fields.get("issuetype", {}),
        "status": fields.get("status", {}),
        "priority": fields.get("priority", {}),
        "labels": fields.get("labels", []),
        "components": fields.get("components", []),
        "story_points": story_points,
        "sprint": sprint_name,
        "assignee": fields.get("assignee", {}),
        "acceptance_criteria": fields.get("customfield_10014") or "",  # varies by org
        "comments": comments,
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def format_narrative_md(key: str, narrative: PMNarrative) -> str:
    """Render a PMNarrative as clean Markdown."""
    confidence_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(narrative.confidence, "⚪")
    lines = [
        f"# {key} — PM Analysis",
        f"*Confidence: {confidence_emoji} {narrative.confidence}*",
        "",
        "## Executive Summary",
        narrative.executive_summary,
        "",
        "## Business Impact",
        narrative.business_impact,
        "",
        "## User Implications",
        narrative.user_implications,
        "",
        "## Leadership Evaluation",
        narrative.leadership_evaluation,
        "",
        "## Strategic Signals",
        narrative.strategic_signals,
        "",
        "## PM Follow-up Questions",
    ]
    for i, q in enumerate(narrative.pm_questions, 1):
        lines.append(f"{i}. {q}")

    if narrative.missing_context:
        lines += [
            "",
            "## What Would Sharpen This Analysis",
            narrative.missing_context,
        ]

    return "\n".join(lines)


def _write_output(text: str, path: Optional[str]) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Written to {path}", file=sys.stderr)
    else:
        print(text)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Jira data infrastructure stories into PM narratives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze subcommand ───────────────────────────────────────────────────
    analyze_cmd = subparsers.add_parser(
        "analyze",
        help="Analyze issue(s) from a JSON file or stdin",
    )
    analyze_cmd.add_argument(
        "input",
        help="Path to JSON file with a single issue or a list of issues, or '-' for stdin",
    )
    analyze_cmd.add_argument("--output", "-o", help="Output Markdown file (default: stdout)")
    analyze_cmd.add_argument("--verbose", "-v", action="store_true", help="Show thinking preview")

    # ── fetch subcommand ─────────────────────────────────────────────────────
    fetch_cmd = subparsers.add_parser(
        "fetch",
        help="Fetch from Jira API and analyze",
    )
    fetch_cmd.add_argument("issue_key", help="Jira issue key, e.g. DATA-123")
    fetch_cmd.add_argument(
        "--jira-url",
        default=os.environ.get("JIRA_URL"),
        help="Jira base URL (or set JIRA_URL)",
    )
    fetch_cmd.add_argument(
        "--jira-email",
        default=os.environ.get("JIRA_EMAIL"),
        help="Jira account email (or set JIRA_EMAIL)",
    )
    fetch_cmd.add_argument(
        "--jira-token",
        default=os.environ.get("JIRA_API_TOKEN"),
        help="Jira API token (or set JIRA_API_TOKEN)",
    )
    fetch_cmd.add_argument("--output", "-o")
    fetch_cmd.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    client = anthropic.Anthropic()

    # ── analyze ──────────────────────────────────────────────────────────────
    if args.command == "analyze":
        raw = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
        try:
            data = json.load(raw)
        finally:
            if args.input != "-":
                raw.close()

        issues = data if isinstance(data, list) else [data]
        print(f"Analyzing {len(issues)} issue(s)…", file=sys.stderr)

        results = batch_analyze(issues, client=client, verbose=args.verbose)

        sections = []
        for r in results:
            if "narrative" in r:
                sections.append(format_narrative_md(r["key"], r["narrative"]))
            else:
                sections.append(f"# {r['key']} — Error\n\n{r['error']}")

        _write_output("\n\n---\n\n".join(sections), args.output)

    # ── fetch ────────────────────────────────────────────────────────────────
    elif args.command == "fetch":
        missing = [
            name
            for name, val in [
                ("JIRA_URL / --jira-url", args.jira_url),
                ("JIRA_EMAIL / --jira-email", args.jira_email),
                ("JIRA_API_TOKEN / --jira-token", args.jira_token),
            ]
            if not val
        ]
        if missing:
            print(f"Error: missing required values: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

        print(f"Fetching {args.issue_key} from Jira…", file=sys.stderr)
        issue = fetch_from_jira(
            args.issue_key, args.jira_url, args.jira_email, args.jira_token
        )
        print("Analyzing…", file=sys.stderr)
        narrative = analyze_story(issue, client=client, verbose=args.verbose)
        _write_output(format_narrative_md(args.issue_key, narrative), args.output)


if __name__ == "__main__":
    main()
