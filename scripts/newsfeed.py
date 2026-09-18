import anthropic
import httpx
import json
import hashlib
import nh3
import re
import smtplib
import os
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# The report is built from live web-search results, which are untrusted input.
# A malicious page can attempt indirect prompt injection to make the model emit
# tracking beacons (<img>), scripts, or javascript:/data: links that would fire
# or exfiltrate when the email is opened. We never trust the model output as
# safe HTML — it is run through an allowlist sanitizer before being emailed.
# Only these tags/attributes survive; everything else (img, script, style,
# iframe, event handlers, non-http(s)/mailto URLs) is stripped.
#
# `span` and `class` are allowed so the model can mark up structural hooks
# (item cards, tier headers) that our trusted stylesheet targets. They are
# cosmetic only and don't widen the security surface: `class` cannot execute
# or exfiltrate, and the `style` attribute, `<style>` element, scripts,
# images, and event handlers all remain stripped. Worst case from an injected
# class is a misplaced tier header — a visual nuisance, not a vulnerability.
ALLOWED_TAGS = {
    "h2", "h3", "p", "strong", "em", "ul", "ol", "li", "div", "a", "br", "span",
}
_CLASS_ONLY = {"class"}
ALLOWED_ATTRIBUTES = {
    "a": {"href", "title", "class"},
    "div": _CLASS_ONLY,
    "span": _CLASS_ONLY,
    "p": _CLASS_ONLY,
    "h2": _CLASS_ONLY,
    "h3": _CLASS_ONLY,
    "ul": _CLASS_ONLY,
    "ol": _CLASS_ONLY,
    "li": _CLASS_ONLY,
    "strong": _CLASS_ONLY,
    "em": _CLASS_ONLY,
}
ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


def sanitize_html(html):
    return nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        url_schemes=ALLOWED_URL_SCHEMES,
    )


# Example profile only. The real candidate profile is injected at run time via
# the CANDIDATE_PROFILE environment variable (a GitHub Actions secret) so
# personal details never live in the repository. A custom profile must keep
# this shape — a "WHO THE CANDIDATE IS" section (background, target roles and
# verticals, home region, remote/hybrid/onsite preference) followed by a
# "WATCHLIST COMPANIES" section ending in the company list — because the
# prompt text that follows it refers back to both (see the HOME REGION block
# in the prompt below).
DEFAULT_PROFILE = """WHO THE CANDIDATE IS

The candidate is pursuing three distinct but related role identities, each with a different expected share of the opportunities that will actually be available. Identity 1 (~50% of expected opportunities): Director of Strategic Initiatives, Transformation Director, or Chief of Staff, at mission-driven organizations or consulting firms with a social-impact practice. Identity 2 (~30%): Market Intelligence Director or Competitive Intelligence Director, at retail, insights agencies, or consumer products companies. Identity 3 (~20%): CX Insights Director or Experience Strategy Director, at similar organizations with a customer-experience focus.

The candidate is based in Chicago, IL. Onsite or hybrid roles are a match only if realistically reachable within about a 1-hour public-transit commute from the Loop or Fulton Market. Fully remote roles are a match regardless of company location.

---

WATCHLIST COMPANIES

Bridgespan Group, Circana, NielsenIQ, Gartner, American Heart Association."""


# Approximate published Opus 4.8 rates ($5/$25 per MTok), in USD per token.
# Adjust if pricing changes — these only drive the logged cost estimate, not
# anything functional.
PRICE_INPUT = 5 / 1_000_000           # fresh (uncached) input
PRICE_CACHE_WRITE = 6.25 / 1_000_000   # cache creation = 1.25x input
PRICE_CACHE_READ = 0.5 / 1_000_000     # cache read = 0.1x input
PRICE_OUTPUT = 25 / 1_000_000
PRICE_WEB_SEARCH = 10 / 1_000          # $10 per 1,000 searches


def accumulate_usage(totals, usage):
    """Add one API response's usage onto the running totals for the run."""
    server_tool = getattr(usage, "server_tool_use", None)
    totals["input"] += getattr(usage, "input_tokens", 0) or 0
    totals["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    totals["output"] += getattr(usage, "output_tokens", 0) or 0
    totals["searches"] += (
        getattr(server_tool, "web_search_requests", 0) or 0 if server_tool else 0
    )


def log_usage(totals):
    """Print run-total token/search usage and an estimated dollar cost."""
    est_cost = (
        totals["input"] * PRICE_INPUT
        + totals["cache_write"] * PRICE_CACHE_WRITE
        + totals["cache_read"] * PRICE_CACHE_READ
        + totals["output"] * PRICE_OUTPUT
        + totals["searches"] * PRICE_WEB_SEARCH
    )

    print(
        f"Usage across {totals['api_calls']} API call(s) — "
        f"input(fresh): {totals['input']:,}, cache write: {totals['cache_write']:,}, "
        f"cache read: {totals['cache_read']:,}, output: {totals['output']:,}, "
        f"web searches: {totals['searches']:,}\n"
        f"Estimated cost: ${est_cost:.2f} "
        "(rate estimate; verify against Anthropic pricing)"
    )


# Upper bound on pause_turn continuations (the server-side tool loop pauses
# roughly every 10 tool iterations); a guard against a runaway loop, not a
# budget — search spend is capped by max_uses on the web_search tool. Raised
# slightly above the single-identity baseline of 12: watchlist/named-org
# checks (up to 10), three separate per-identity ATS-wide sweeps across six
# job-board domains each, aggregator checks, transformation-signal screening,
# and a live web_fetch verification per candidate role can add up to more
# tool iterations than the search cap alone suggests, given the ATS sweep now
# runs three times per scan instead of once. (A Greenhouse/Workday API
# cross-check was tried and removed as non-functional — see the KNOWN
# RESIDUAL LIMITATION note above — so this value doesn't need to account for
# that.)
MAX_PAUSE_CONTINUATIONS = 14

# Full-scan retries when the streaming connection dies mid-read ("peer closed
# connection", read timeout). The SDK's max_retries doesn't cover these — it
# only retries failed request setup — and a dropped stream loses the whole
# in-flight report, so the only recovery is to start the scan over.
STREAM_RETRIES = 2


# --- Cross-run deduplication -------------------------------------------------
#
# The model re-verifies every role fresh on every run — that behavior is
# unchanged. This layer runs AFTER the model has produced its report and only
# affects presentation: a role the candidate has already seen in a previous
# report gets collapsed to a one-line "still open" note instead of repeating
# its full write-up, while a genuinely new role keeps full detail exactly as
# before. This still confirms liveness every week; it just stops re-showing
# the same paragraph for a role that's been open for six weeks running.
#
# PRIVACY NOTE: this repo is public, and the rest of this script deliberately
# persists nothing (see the INVARIANT comment near __main__). Making dedup
# work at all requires SOME state to survive between runs, so this is a
# narrow, intentional exception — but we store only a one-way hash of each
# role's URL plus a first-seen date, never the plaintext company, title, or
# URL. A hash lets the script recognize "I've seen this exact URL before"
# without leaving a human-readable history of the candidate's search in git.
STATE_FILE = "state/seen_roles.json"

ITEM_RE = re.compile(r'<div class="item">.*?</div>', re.DOTALL)
SOURCE_HREF_RE = re.compile(r'Source:.*?href="([^"]+)"', re.DOTALL)
TITLE_RE = re.compile(r'<h3[^>]*>(.*?)</h3>', re.DOTALL)


def hash_url(url):
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def load_seen_roles():
    """Return {url_hash: first_seen_date} from disk, or {} if absent/corrupt.

    Corrupt or missing state is never fatal — worst case, dedup silently does
    nothing this run (every role looks "new" again), which is a presentation
    regression, not a broken report.
    """
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return {e["h"]: e["first_seen"] for e in entries}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(
            f"State file unreadable ({type(exc).__name__}); "
            "proceeding without cross-run dedup this run.",
            file=sys.stderr,
        )
        return {}


def save_seen_roles(state):
    """Write {url_hash: first_seen_date} back to disk as a sorted JSON list.

    Sorted by hash for a stable, minimal diff each week (only genuinely added
    or dropped entries change). Entries not present in this run's confirmed
    set are simply omitted here, which naturally prunes roles that closed or
    weren't rediscovered — no separate age-based cleanup needed.
    """
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    entries = [
        {"h": h, "first_seen": first_seen}
        for h, first_seen in sorted(state.items())
    ]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")


def dedupe_against_history(report_html):
    """Collapse previously-seen roles to a one-line note; keep new ones full.

    Operates on the already-sanitized report fragment. IMPORTANT: every role
    in the highlights box at the top is intentionally duplicated into a tier
    below it (that's by design, not a bug — a prior request specifically kept
    this). Only the tier content is scanned for dedup; the highlights box is
    left completely untouched. Two reasons: (1) if both copies of a repeat
    role were deduped independently, the same role produces two identical
    "still open" lines — a real bug fixed here; (2) stripping a repeat out of
    highlights but not out of its tier left the highlights box visibly
    truncated (e.g. "Top 3" showing only one role) even though nothing was
    actually wrong. The highlights box now always shows whatever the model
    picked, full detail, same as every previous report.

    Every <div class="item"> block from the first tier header onward is
    checked by its Source link's URL hash against the persisted state. A
    match means the candidate has already seen this exact posting in a prior
    report, so the full item is removed from its tier and a compact line is
    appended in a new "STILL OPEN — NO CHANGE" section instead. A miss means
    it's new: left in place untouched, and recorded for next time.

    Wrapped in a broad try/except by the caller — a bug here should degrade to
    "no dedup this run," never block the report from sending.
    """
    seen_state = load_seen_roles()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_state = {}
    repeat_entries = []
    seen_hashes_this_run = set()

    # Split at the first tier header so the highlights box above it is never
    # scanned or modified. Per FORMAT AND MARKUP, highlights always precedes
    # both tier headers, so this boundary is reliable. If no tier header is
    # found (e.g. "Nothing confirmed this week"), there's nothing to dedupe —
    # process the whole thing as a no-op pass-through.
    tier_split = report_html.find('<h3 class="tier">')
    if tier_split == -1:
        head, tail = report_html, ""
    else:
        head, tail = report_html[:tier_split], report_html[tier_split:]

    def replace_item(match):
        item_html = match.group(0)
        href_match = SOURCE_HREF_RE.search(item_html)
        if not href_match:
            # No identifiable source link — can't dedupe it, leave as-is and
            # don't track it (better to show an extra item than lose one).
            return item_html

        url = href_match.group(1).strip()
        h = hash_url(url)
        title_match = TITLE_RE.search(item_html)
        title_text = title_match.group(1).strip() if title_match else "Untitled role"

        if h in seen_state:
            new_state[h] = seen_state[h]  # carry forward the original date
            if h not in seen_hashes_this_run:
                seen_hashes_this_run.add(h)
                repeat_entries.append((title_text, url, seen_state[h]))
            return ""  # dropped from its tier; listed compactly below instead
        else:
            new_state[h] = today_str
            return item_html  # genuinely new — keep in place, full detail

    deduped_tail = ITEM_RE.sub(replace_item, tail)
    deduped_html = head + deduped_tail

    if repeat_entries:
        compact_items = "\n".join(
            f'<li><a href="{url}">{title}</a> — still open, first seen {first_seen}</li>'
            for title, url, first_seen in repeat_entries
        )
        deduped_html += (
            "<h2>STILL OPEN — NO CHANGE</h2>"
            "<p>These roles appeared in a previous report and were re-verified "
            "live again this run; full details aren't repeated since nothing "
            "changed. Click through for the full posting.</p>"
            f"<ul>{compact_items}</ul>"
        )

    save_seen_roles(new_state)
    # Defense in depth: the pieces above are all built from already-sanitized
    # fragments, but a second pass is cheap and free of surprises.
    return sanitize_html(deduped_html)


def get_newsfeed():
    # Weekly cadence: one transient 529/5xx would otherwise cost a whole week,
    # so retry harder than the SDK default of 2.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=4)

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    profile = os.environ.get("CANDIDATE_PROFILE", "").strip() or DEFAULT_PROFILE

    prompt = f"""Today's date is {today}.

You must use live web search for every item in this report. Do not rely on your training data for any factual claim about a company or a role. If you cannot find a live, dated source for an item, do not include it. You also have a web_fetch tool: use it to open every candidate job posting directly and confirm from the fetched page — never from a search snippet — that the role is still live and open to applications.

You are a research assistant supporting a Strategy, Insights, and Customer Experience executive — referred to throughout as "the candidate" — who is actively searching for a Director-to-VP level role across three related but distinct role identities, described below.

---

{profile}

This watchlist is a starting point, not a boundary. The search is profile-driven, not list-driven: any organization matching one of the three identities' own target industries is in scope regardless of whether it appears above. Expect most of the best findings each week to come from organizations NOT on the watchlist.

---

OPEN ROLES SCAN
HOME REGION. The candidate is based in Chicago, IL. Onsite or hybrid roles count as a location match only if they are realistically reachable within about a 1-hour public-transit commute from the Loop or Fulton Market — do not treat every posting tagged "Chicago" as a match; a car-dependent suburban office park is not, even if the posting's city field says Chicago or a Chicago suburb. A fully remote role counts as a location match regardless of company location. When a posting's specific office location isn't stated precisely enough to judge transit reachability, note that uncertainty rather than assuming a match.

TARGET IDENTITIES. The candidate is pursuing three distinct but related role identities, each with a different expected share of the opportunities that will actually be available, and each defined by its own titles AND its own target industries — do not treat this as one undifferentiated title family, and do not apply one identity's industry list to another identity's titles.

IDENTITY 1 — STRATEGY & TRANSFORMATION LEADER (~50% of expected opportunities; the primary search focus, and the tie-breaker when ranking comparably strong roles across identities)
Titles: Director of Strategic Initiatives, Transformation Director, Chief of Staff, and close variants such as Head of Strategy & Transformation, VP of Strategic Initiatives, or Director/Head of the Chief of Staff Office.
Target organizations: mission-driven organizations scaling up or undergoing transformation; "responsible brands" (a genuine public sustainability, ESG, or social-responsibility positioning — not just marketing language, use judgment); consulting firms with a social-impact focus (e.g. Bridgespan Group, Civic Consulting Alliance, FSG) or traditional consulting firms with a public-sector, nonprofit, or social-impact practice.
Priority sector: healthcare and life sciences — hospitals and health systems, public health and community health organizations, healthcare technology, and healthcare nonprofits (e.g. American Heart Association, Alzheimer's Association).
Exploratory sectors (include if found, but flag as exploratory — these are lower-confidence than the priority sector above): education technology, civic technology, AI.

IDENTITY 2 — INTELLIGENCE & INSIGHTS LEADER (~30% of expected opportunities)
Titles: Market Intelligence Director, Competitive Intelligence Director, and close variants such as Head of Market Intelligence or Director of Competitive Insights.
Target organizations, in rough priority order: retail (preferably responsible brands); insights agencies (e.g. Circana, NielsenIQ, Kantar, Ipsos, Gartner); brand agencies; consumer products companies (preferably responsible brands); healthcare and life sciences, excluding insurance.

IDENTITY 3 — EXPERIENCE & CUSTOMER STRATEGY LEADER (~20% of expected opportunities)
Titles: CX Insights Director, Experience Strategy Director, and close variants such as Head of Customer Experience Strategy or Director of CX Insights.
Target organizations, in rough priority order: retail (preferably responsible brands); insights agencies (same list as Identity 2); brand agencies; consumer products companies (preferably responsible brands); consulting firms, most likely those with a digital CX transformation practice; healthcare and life sciences, excluding insurance.

For every confirmed role, state which identity it matches. If a role plausibly fits more than one identity, note that too — it's a positive signal, not something to resolve by picking just one.

DISCOVERY STRATEGY. The search is profile-driven: most qualifying roles each week will be at organizations not on the watchlist, so do not simply iterate the watchlist company by company. Run these discovery passes, in this order:

1. Watchlist companies: check the careers pages of the watchlist companies above for openings matching any of the three identities' titles.
2. Named-organization checks: beyond the formal watchlist, directly check the careers pages of the specific organizations named in the identity descriptions above (e.g. Bridgespan Group, Civic Consulting Alliance, FSG for Identity 1; Circana, NielsenIQ, Kantar, Ipsos, Gartner for Identities 2 and 3), since these are known-relevant targets even though they weren't formalized into the watchlist.
3. ATS-wide title sweeps, run separately per identity so results aren't diluted — for example: site:boards.greenhouse.io "Director" "Strategic Initiatives" for Identity 1, site:boards.greenhouse.io "Director" "Market Intelligence" for Identity 2, site:boards.greenhouse.io "Director" "Customer Experience" for Identity 3 — across the major applicant-tracking-system domains: boards.greenhouse.io, jobs.lever.co, jobs.ashbyhq.com, myworkdayjobs.com, jobs.smartrecruiters.com, apply.workable.com. This is the highest-yield way to find organizations the candidate has never heard of. Filter each identity's hits to organizations matching that identity's own target industries — not another identity's.
4. Job aggregators for discovery: LinkedIn Jobs, Built In Chicago and remote, Wellfound, and Welcome to the Jungle/Otta. Aggregators are for discovery only — always follow through to the underlying organization's posting and cite that as the Source, never the aggregator page.
5. Transformation-signal screening: search for organizations in the candidate's target industries with a recent, dated signal that predicts a Strategy/Transformation, Chief of Staff, Intelligence, or CX leadership hire — a new CEO or executive transition, a publicly announced strategic plan or transformation initiative, or a major funding round or scaling announcement at a mission-driven organization. Check those organizations' careers pages and the ATS domains above for their openings specifically. Weight this pass toward Identity 1's industries, since this signal maps most directly to that identity's titles.

Aim for breadth of organizations over exhaustive depth on any one. A weekly report that surfaces 8 to 15 verified roles across many organizations is more useful than 3 roles from the watchlist plus an exhausted search budget.

PRESENT ROLES IN TWO TIERS. Search broadly, but do not present the results as one flat list — breadth is valuable for discovery but creates noise when every role is shown with equal weight. Split the confirmed roles into two labeled sub-sections, strongest first within each:

- "Strong fits": roles where title/function AND target industries clearly match one identity's own definition (seniority: Director, Senior Director, VP, Head of, or Chief level), and the location matches the candidate's home region as defined above (Chicago within transit reach, or unrestricted remote). These are the roles they should look at first. A role can be a Strong fit even if it's "staleness/completeness unconfirmed" per the rule below — that status affects what you caveat, not which tier it belongs in. When ranking within this tier, an Identity 1 role edges out an equally strong Identity 2 or 3 role, reflecting its larger expected share of opportunities.
- "Broader — worth a look": real, verified, currently-live roles that are a stretch on one dimension — seniority slightly off, industry adjacent to an identity's list rather than core, title matching one identity but industry matching another (a genuine cross-identity stretch, not an error), location outside the home region, or an onsite/hybrid role whose transit reachability from the Loop or Fulton Market couldn't be confirmed. Include these for discovery value, but cap this tier at the 8 strongest; if more than 8 qualify, keep the 8 best fits to the candidate's profile and drop the rest rather than padding the list.

Do not relax the CORE liveness bar for either tier — a role must still be fetched and show its exact title, description, and an active apply control, with no closed/expired/redirect/error signal, to appear in either. That bar is unchanged. What has changed is that a missing or stale date, or a missing secondary field, no longer forces exclusion on its own — see "staleness/completeness unconfirmed" below. Tiering is about ranking what you found, never about lowering the CORE bar for what counts as verified. If a role is a genuine strong fit, it goes in "Strong fits" even if it is the only role this week.

There is no recency window on this scan — roles are governed by whether they are currently live, not by when they were first posted. A still-open role posted three weeks ago is in scope; a role posted yesterday that has already closed is not.

Every posting you include must be currently live and open to applications. Search results and search-engine snippets routinely surface roles that have already been filled or closed, so a search hit is not sufficient evidence that a role is open. Before including any role, open the posting page itself with web_fetch and confirm from the fetched content that it is still accepting applications. A role you did not fetch does not go in the report.

A page that returns successfully is NOT proof the role is live. Closed postings very frequently still "work" but silently redirect to the company's default careers homepage, a job-search index, or a generic "open positions" listing, while the original link continues to resolve. Some companies instead serve a branded error page at the dead URL — their normal site header, navigation, and styling intact, with an illustration and message like "Oops, let's fix this" or "Job not found," rather than a plain HTTP error or an obvious redirect. This is just as dead as any other broken link — do not treat a page that merely looks polished and on-brand as evidence the role exists; a company's design system stays consistent whether the specific page behind it is a real job or an error state. You must confirm that the final page you land on actually displays that exact role — its specific title and description, with an active apply control. If the link instead lands on a careers homepage, a job-search or "open positions" index, a search results page, a branded or unbranded error page, or a "job not found" / "this position is no longer available" page, the role is dead — exclude it. The link you put in the Source field must point to that live, role-specific detail page, not to a redirect target, error page, or careers landing page.

Grounding check before including any role: you must be able to point to the exact sentence(s) in the content you fetched — not the URL, not the search snippet, not what a posting at this company usually looks like — that state the role's title and show an active apply control. If you cannot identify that specific fetched text, do not include the role, and do not fill in plausible-sounding details (compensation bands, location list, posting date) from general knowledge of what this company's postings typically look like. Every field you report for a role — title, compensation, locations, posting date — must come from content you actually fetched for that specific posting, not from inference about the company or the role type.

The active apply control specifically must be DIRECTLY observed, never inferred. Seeing an applicant-privacy notice, an EEO disclosure, or similar boilerplate that would typically accompany an application form is NOT the same as seeing the actual apply control itself, and does not satisfy this requirement on its own — plenty of dead postings still show that boilerplate text. If the fetched content doesn't let you point to the actual apply button, link, or form fields themselves, the role fails the CORE liveness bar and must be hard-rejected — this is never eligible for the "staleness/completeness unconfirmed" treatment below, which only ever applies to a missing date or a missing secondary field, and never to the apply control itself.

Reject the posting outright — do not list it in any form — only for HARD evidence the specific role is gone: the page does not load or returns an error; the link redirects to or lands on a generic careers page, job-search index, or listing rather than the specific role's detail page; the page is a branded or unbranded error/not-found page, even one styled consistently with the rest of the company's site; the final page does not display that exact role's title and description with an active apply control; or the page states the role is closed, filled, paused, on hold, expired, or "no longer accepting applications." These are the only outcomes that mean the role is actually dead. When any of these are true, exclude rather than guess: an omitted role is fine, a dead role is the failure mode to avoid.

KNOWN RESIDUAL LIMITATION, accepted rather than solved: on some platforms (confirmed on Greenhouse; suspected on others), the rendered page can display a complete, convincing, fully-populated posting — full description, comp, an apparently-active apply form — for a role that has actually closed, because true accept/reject status is determined by a live client-side check a fetch tool cannot execute. An API-based cross-check was attempted and removed: the web_fetch tool used here can only reach URLs that already appear in a prior search or fetch result, which a constructed API endpoint never does, so the check could never actually run. There is no known fix for this within the current toolset. It is mitigated, not solved, by the report's standing instruction to the candidate to manually verify via a private browser window before applying — do not attempt to re-add a platform-specific API check without first confirming, with a real test, that the tool can actually reach that endpoint.

Do NOT add to that hard-reject list: a missing or absent posting/last-refreshed date, a posting date more than 30 days old, or a missing secondary field like location when everything else about the posting (exact title, description, active apply control, no closed/expired signal) is intact. None of these are evidence the role is dead — a senior-level search can legitimately stay open for months, and some ATS platforms display an on-page "posted X days ago" label that never actually updates (i.e. it can read the same "2 days ago" for months — verified directly on this pipeline), making it actively unreliable as a freshness signal, not just an absent one. Treat a role like this as a third outcome, distinct from both confirmed-live and confirmed-dead: CONFIRMED LIVE, STALENESS/COMPLETENESS UNCONFIRMED. Include it in whichever tier its title/vertical/location otherwise earn — do not exclude it and do not route it to the manual check list (that list is specifically for pages the fetch tool couldn't render at all; this is a page that rendered completely but has an uncertain or missing date, or a missing non-essential field). Instead, state plainly in the Fit field exactly what's uncertain (e.g. "posting shows no date — could not confirm freshness beyond the active apply control" or "location not stated on the fetched page") so the candidate knows precisely what to double-check, the same way an undisclosed compensation range is already stated as "Not disclosed on posting" rather than causing exclusion. The grounding-check rule still applies in full: never invent a plausible-sounding date, location, or other field — state the uncertainty instead of guessing.

For each confirmed role, provide the role title, company, and a direct link to the role-specific posting itself (not a search results page, careers homepage, or job-aggregator listing). Report the posting or last-refreshed date exactly as it appears on the page; if no date is shown, or the only available date signal is stale/unreliable (see above), say so plainly rather than omitting the field or guessing — this is a "staleness/completeness unconfirmed" case, not a reason to exclude the role. While you have the posting open to verify it is live, also capture the stated compensation range if one is present — US postings frequently disclose it under pay-transparency laws — and report it exactly as written; if the page doesn't show location, compensation, or another expected field, state "Not stated on posting" for that field rather than excluding the role over it. If the company is in IPO preparation or publicly known to be approaching IPO within 18 months, flag this prominently — it is a high-priority hiring signal. If you cannot confirm a single live role this week, output: "Nothing confirmed this week."

For every confirmed role, note its location and work arrangement (remote, hybrid, or onsite) as stated on the posting. If the role does not match the candidate's home region as defined above, additionally flag the organization's current work-location posture: whether it has recently announced or enforced a significant Return-to-Office (RTO) mandate, or whether it is genuinely remote-friendly. Base this on dated, verifiable sources — the posting's own remote/location terms, an organization's announcement, or recent news coverage — and say so briefly if you cannot confirm either way. This flag is informational only: do NOT exclude, downrank, or filter out an otherwise relevant out-of-region role because of an RTO push or because the work arrangement is unclear. The candidate still wants to see these roles; the flag simply tells them what they would be walking into. Roles matching the candidate's home region do not need the RTO research.

---

SEARCH BUDGET

Spend the search budget on the discovery passes above, in the order listed — watchlist checks first, then the ATS-wide sweep (the highest-yield pass), then aggregators, then IPO-pipeline screening. Live-verification of individual postings is done with web_fetch, which does not draw from this search budget.

---

OUTPUT FORMAT

Begin your response with the opening HTML tag. Do not narrate your search process, describe your methodology, summarize what you are about to do, or include any preamble or transitional language before the HTML output. The report starts with the HTML — nothing before it.

At the top of the report, flag the three strongest roles overall (by fit to the candidate's profile, with Identity 1 breaking ties against equally strong Identity 2 or 3 roles, and named watchlist/named-organization involvement breaking any remaining ties), wrapped in <div class="highlights">…</div>, using the same item fields as below.

For each role, provide:
- Identity: which of the three target identities this role matches (Strategy & Transformation, Intelligence & Insights, or Experience & Customer Strategy); note a second identity too if the role plausibly fits more than one.
- What happened: one to two sentences, factual and specific.
- Why it matters to the candidate: one to two sentences on how this fits the job search.
- Recommended action: a specific next step and, where relevant, a time window.
- Fit: one sentence stating why this role fits the candidate's profile and the single biggest caveat or stretch (e.g. "Core Chief of Staff mandate at a mission-driven healthcare nonprofit; caveat: onsite with an unclear transit-reachable office location"). If the role is "staleness/completeness unconfirmed" (see verification rules above), that IS the caveat to lead with here (e.g. "caveat: posting shows no date, so freshness beyond the active apply control could not be confirmed"), even if there's also a location/seniority stretch — the candidate needs to know to double-check this one before investing time. This is what lets the candidate skim-accept or skim-reject in one read.
- Location and work arrangement: the role's location and whether it is remote, hybrid, or onsite. For roles outside the candidate's home region, also flag whether the company has a recent Return-to-Office (RTO) push or is remote-friendly, with the basis for that flag. This is informational and never a reason to omit the role.
- Transformation signal (if applicable): any dated, verifiable signal that predicted this opening — a new CEO or leadership transition, an announced strategic plan, or a major funding/scaling announcement — with its basis. Omit this field entirely when no such signal was found; it's a bonus, not a requirement.
- Compensation: the pay range exactly as stated on the posting you fetched, including what it covers (base, on-target earnings, bonus, equity) if specified — e.g. "$180K–$220K base + bonus." If the posting shows no range, write "Not disclosed on posting"; you may add a market estimate ONLY if you find a dated, citable public source and label it clearly as an estimate with that source. Never invent or guess a number from general knowledge.
- Source: direct link to the role-specific posting itself.

When flagging errors and limitations, apply the following rules throughout the report, and track them for the COVERAGE NOTES section below rather than only mentioning them inline.
If a job posting cannot be confirmed as currently live and accepting applications by opening the posting page, exclude it entirely. Do not list unverified or stale roles even with a caveat — in this report a wrong listing is worse than an omission.
If web search returns no results for a specific watchlist company, do not infer absence of openings. Note it as: "No confirmed results found for [company] this week — coverage may be incomplete."

Distinguish WHY a posting was excluded, because the reasons carry different meaning for the candidate:
- Confirmed dead: the fetched page explicitly shows the role is closed, filled, expired, or redirects to a generic careers/search page — real evidence the specific role is gone. Do not add these to the manual check list; the exclusion itself is the useful information.
- Fetch could not render the content: the page returned successfully but the fetched content is empty, near-empty, or generic boilerplate with no role-specific title/description/apply-control visible — the signature of a client-side-rendered (JavaScript-heavy) page rather than a confirmed-dead one. This is a tooling limitation, not evidence the role is gone — a real, live role may be sitting behind that exact page. Track every organization/platform this happens for.
- Staleness/completeness unconfirmed is NOT an exclusion category at all — it's not one of the two buckets above. A role with a missing/stale date or a missing secondary field, but a fully rendered page and a real, active apply control, is included in the main report (Strong fits or Broader, per its own merits) with the uncertainty stated in its Fit field, per the verification rules above. Do not list it here in COVERAGE NOTES and do not also list it in the main report — it belongs in the main report only.

MANUAL CHECK LIST. Compile every organization from this run that hit the "fetch could not render the content" case above into a single deduplicated list (one entry per organization, even if it happened on multiple postings or platforms for that organization this week). For each entry, give: the organization's name, the platform/domain where this happened if identifiable (e.g. Workday, Ashby, or the organization's own site), and a direct link to that organization's general careers/job-search landing page (not the specific unrenderable posting, since that's exactly the link that couldn't be verified) so the candidate can check it by hand. If no organizations hit this case this week, state "No fetch-rendering issues this week" rather than omitting the section.

FORMAT AND MARKUP

Output ONLY the report body as an HTML fragment. Do NOT include <!doctype>, <html>, <head>, <body>, <style>, or any CSS — a styling shell is wrapped around your output automatically. Do not set any colors, fonts, or style attributes yourself; the only styling you control is the class names listed below, which hook into that shell. Do not invent other class names or use any class not listed here.

Structure:
- <h2> for the report's single section header (e.g. "OPEN ROLES").
- <h3> for item titles (the role title + company).
- Wrap every individual item in <div class="item">…</div>.
- <strong> for field labels (e.g. <strong>Why it matters to the candidate:</strong>).
- Plain prose in <p> tags; lists in <ul>/<li>. Use <a href="…"> for every source link.
- For the three highest-priority roles at the very top, wrap that whole block in <div class="highlights">…</div>.
- Render the two tier sub-headers as <h3 class="tier">Strong fits</h3> and <h3 class="tier">Broader — worth a look</h3>, each followed by that tier's item divs.
- After the two tiers, add a <h2> "COVERAGE NOTES" section. Within it, use an <h3 class="tier">Manual check list</h3> sub-header followed by a <ul> where each <li> is one company (name, platform if known, and an <a href="…"> to its general careers page). Any other coverage notes (no-results-found companies, general caveats) go in plain <p>/<ul> content in this same COVERAGE NOTES section, above or below the manual check list as makes sense.

Do not use markdown. No inline JavaScript, no images, no tables. Keep nesting shallow and clean.
"""

    totals = {
        "input": 0, "cache_write": 0, "cache_read": 0, "output": 0,
        "searches": 0, "api_calls": 0,
    }

    def run_scan():
        """One full scan: stream the request, following pause_turn continuations.

        The server-side tool loop pauses (stop_reason "pause_turn") after ~10
        tool iterations. Keep continuing the same conversation until the model
        finishes for real. Streaming avoids the SDK's 10-minute non-streaming
        timeout.
        """
        # Cache the large static prompt. The web-search/web-fetch tool loop
        # makes many model turns within this call (and across pause_turn
        # continuations and scan retries); caching means later turns read the
        # prefix from cache at ~10% the cost instead of reprocessing it.
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }],
        }]
        text_parts = []
        message = None
        # The dynamic-filtering web_search/web_fetch tools run their result
        # filtering as server-side code execution inside a container. When the
        # tool loop pauses (pause_turn) with pending code-execution tool uses,
        # the continuation must be pinned to that same container by passing its
        # id back — otherwise the API rejects the resume with "container_id is
        # required when there are pending tool uses generated by code execution
        # with tools." The container id is created mid-turn by code execution,
        # so it first appears on a `message_delta` stream event, never on
        # `message_start`. Crucially, the non-beta stream accumulator behind
        # `client.messages.stream` does NOT copy that container onto the final
        # message (only the beta accumulator does), so `get_final_message()`
        # always reports `container=None` here. We therefore read the id off the
        # `message_delta` events directly as the stream is consumed below.
        container_id = None

        for _ in range(1 + MAX_PAUSE_CONTINUATIONS):
            stream_kwargs = {
                "model": "claude-opus-4-8",
                "max_tokens": 64000,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "tools": [
                    # max_uses caps search spend at PRICE_WEB_SEARCH * max_uses
                    # per run. Fetches are billed only as input tokens.
                    # Sized for a single-region (Chicago), single-category
                    # (roles-only) scan across THREE separate target identities:
                    # up to 10 watchlist/named-organization checks, three
                    # separate ATS-wide sweeps (one per identity, each across 6
                    # job-board domains with a few title variants), aggregator
                    # checks, and transformation-signal screening, plus headroom
                    # for query reformulation. Set higher than a single-identity
                    # scan specifically because the ATS sweep runs three times
                    # instead of once — watch the per-run search count logged
                    # below against this cap and adjust; if it's regularly
                    # landing near the ceiling, a pass (likely transformation-
                    # signal screening, since it runs last) is probably getting
                    # cut off.
                    {"type": "web_search_20260209", "name": "web_search", "max_uses": 100},
                    {"type": "web_fetch_20260209", "name": "web_fetch"},
                ],
                "messages": messages,
            }
            # Resume in the same container across pause_turn continuations.
            if container_id is not None:
                stream_kwargs["container"] = container_id

            with client.messages.stream(**stream_kwargs) as stream:
                # Consume the events ourselves so we can capture the container
                # id from `message_delta` (the accumulated final message drops
                # it — see the note above). Once assigned mid-turn the id is
                # stable, so we only ever overwrite it with a newer non-null
                # value and otherwise carry the last one forward.
                for event in stream:
                    if event.type == "message_delta":
                        event_container = getattr(event.delta, "container", None)
                        if event_container is not None:
                            container_id = event_container.id
                message = stream.get_final_message()

            totals["api_calls"] += 1
            accumulate_usage(totals, message.usage)
            text_parts.extend(
                block.text for block in message.content if block.type == "text"
            )

            if message.stop_reason != "pause_turn":
                break
            # Re-send the conversation with the paused assistant turn appended;
            # the API resumes the tool loop where it left off.
            messages.append({"role": "assistant", "content": message.content})

        return message, text_parts

    # Retry the whole scan if the stream dies mid-read. A dropped attempt's
    # partial output is unusable, but completed calls' usage is already in
    # `totals`, so the final log still reflects what the run actually cost.
    for attempt in range(1 + STREAM_RETRIES):
        try:
            message, text_parts = run_scan()
            break
        except (anthropic.APIConnectionError, httpx.TransportError) as exc:
            if attempt == STREAM_RETRIES:
                log_usage(totals)  # surface what the failed run still cost
                raise
            print(
                f"Stream dropped mid-run ({exc!r}); restarting scan "
                f"(retry {attempt + 1} of {STREAM_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(30 * (attempt + 1))

    log_usage(totals)

    if message.stop_reason == "pause_turn":
        raise ValueError(
            f"Run was still paused after {MAX_PAUSE_CONTINUATIONS} continuations; "
            "report is incomplete. Report not sent."
        )

    # If the model hit the output cap, the report is truncated mid-section.
    # Surface it instead of emailing a half-complete newsletter.
    if message.stop_reason == "max_tokens":
        raise ValueError(
            "Model response was truncated at the max_tokens limit; "
            "raise max_tokens. Report not sent."
        )

    full_text = "\n\n".join(text_parts)

    # During web search the model emits text blocks narrating each search before
    # producing the report. Drop everything before the first HTML tag so only the
    # report itself is emailed.
    match = re.search(
        r"<(?:!doctype|html|head|body|h[1-6]|div|p|ul|ol|table|section)\b",
        full_text,
        re.IGNORECASE,
    )
    if not match:
        # No HTML report was produced (e.g. the model only narrated, or the call
        # returned empty). Fail loudly rather than emailing raw search narration.
        raise ValueError("Model response contained no HTML report; nothing to send.")

    # Sanitize before returning: web-search content is untrusted and the model's
    # output is not a trusted source of safe HTML (see sanitize_html above).
    report = sanitize_html(full_text[match.start():])
    if not report.strip():
        raise ValueError("Report was empty after sanitization; nothing to send.")

    try:
        report = dedupe_against_history(report)
    except Exception as exc:
        # Dedup is a presentation enhancement, not core to the report's
        # validity — a bug here should never block a good report from
        # sending. Fall back to the undeduped (but still fully sanitized and
        # valid) report and let the run proceed.
        print(
            f"Dedup step failed ({type(exc).__name__}); sending report "
            "without cross-run dedup this run.",
            file=sys.stderr,
        )

    return report

# Dark, flat "sage" theme matching the shared design system. Five-color palette:
# bg #0f120d, surface #1d231c, accent/sage #7d9b83, text #e6e4db, strong #ffffff.
# Every other shade here is a precomputed blend of those — no new hues.
#
# The palette is applied as literal hex, NOT as CSS custom properties: :root/
# var() are unsupported in Outlook (Word engine) and unreliable in Gmail, so the
# email keeps its robust two-layer approach — critical colors set inline on the
# wrapper (survive even where a client drops <style>) and the rest from this
# trusted <style> block (Apple Mail fully, Gmail web/app broadly). Flat only:
# no gradients, no shadows — separation comes from borders and surface
# contrast (bg vs surface), per the design rules.
#
# Fonts: Space Grotesk (display: masthead, headings, tier/label lines) and Inter
# (body and all UI) are named in the font stacks with system fallbacks. The email
# makes NO external font request — the whole newsletter avoids outbound calls
# from the message (no beacons/leaks), and clients widely strip webfonts anyway —
# so the typefaces render where a client already has them and fall back cleanly
# otherwise.
#
# Note: the pill/tag styles (signal-type and hiring-window-temperature) from the
# original nine-category design were dropped along with those categories — this
# roles-only scan has no field that needs them, so the CSS stays free of dead
# selectors the model is never instructed to emit.
EMAIL_STYLE = """
  :root { color-scheme: dark; supported-color-schemes: dark; }
  body { margin: 0; padding: 0; background: #0f120d; -webkit-text-size-adjust: 100%; }
  .wrap { background: #0f120d; padding: 24px 12px; }
  .email {
    max-width: 680px; margin: 0 auto;
    background: #0f120d; border: 1px solid #303b31; border-radius: 12px;
    padding: 4px 26px 14px;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #e6e4db; line-height: 1.55; font-size: 15px;
  }
  .masthead { padding: 22px 0 14px; border-bottom: 2px solid #7d9b83; margin-bottom: 8px; }
  .masthead .title { font-family: "Space Grotesk", "Inter", system-ui, sans-serif; font-size: 20px; font-weight: 700; color: #ffffff; letter-spacing: -0.01em; }
  .masthead .title .accent { color: #7d9b83; }
  .masthead .date { font-size: 12px; color: #909089; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
  .email h2 {
    font-family: "Space Grotesk", "Inter", system-ui, sans-serif;
    font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;
    color: #7d9b83; border-left: 4px solid #7d9b83; padding: 7px 0 7px 12px;
    margin: 34px 0 14px;
  }
  .email h3 { font-family: "Space Grotesk", "Inter", system-ui, sans-serif; font-size: 16px; font-weight: 600; color: #ffffff; margin: 0 0 7px; }
  .email h3.tier {
    font-family: "Space Grotesk", "Inter", system-ui, sans-serif;
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; color: #7d9b83;
    margin: 22px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #303b31;
  }
  .email p { margin: 7px 0; }
  .email strong { color: #aeaea6; font-weight: 600; }
  .email a { color: #7d9b83; text-decoration: none; border-bottom: 1px solid #415143; }
  .email ul, .email ol { margin: 7px 0; padding-left: 20px; }
  .email li { margin: 4px 0; }
  .item {
    background: #1d231c; border: 1px solid #303b31; border-radius: 8px;
    padding: 14px 16px; margin: 0 0 14px;
  }
  .highlights {
    background: #1d231c;
    border: 1px solid #7d9b83; border-radius: 12px; padding: 16px 18px; margin: 16px 0 24px;
  }
  .highlights h3 { color: #ffffff; }
  .verify-banner {
    background: #1d231c; border: 1px solid #b08c4f; border-left: 4px solid #b08c4f;
    border-radius: 8px; padding: 14px 16px; margin: 4px 0 20px;
  }
  .verify-banner .heading {
    font-family: "Space Grotesk", "Inter", system-ui, sans-serif;
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em;
    color: #b08c4f; margin-bottom: 8px;
  }
  .verify-banner ol { margin: 6px 0 0; padding-left: 20px; }
  .verify-banner li { margin: 4px 0; }
  .footer { margin-top: 26px; padding-top: 14px; border-top: 1px solid #303b31; color: #909089; font-size: 12px; }
"""

# Fixed, always-present verification guidance — implemented here rather than
# left to the model to write, so it can never be dropped, shortened, or
# paraphrased away under token pressure or model variance. Every role in the
# report has already been through automated fetch-based verification, but
# that verification has confirmed, real gaps (see the Greenhouse/Workday
# checks in the prompt above) where a fully-populated, legitimate-looking
# page turned out to already be closed. This banner gives the candidate a
# concrete, fast manual step to catch what automation may have missed, rather
# than just a vague "verify before applying" disclaimer.
VERIFY_BANNER = """<div class="verify-banner">
<div class="heading">Before you apply — a 30-second check</div>
<p>Every role below was checked by an automated tool, not a human. Some job platforms keep showing a full, convincing listing for weeks after a role has actually closed — automated verification catches most of these, but not all of them.</p>
<ol>
<li>Open the role's Source link in a <strong>private/incognito browser window</strong> (this avoids any cached or logged-in state showing you an outdated view).</li>
<li>If the page says "no longer active," "this position has been filled," or shows a generic error/careers page instead of the specific role, it's closed — skip it.</li>
<li>For extra caution on <strong>Workday-hosted roles</strong> (links containing myworkdayjobs.com), also search the company's own careers page directly for the role title, since Workday postings are the hardest for this tool to verify automatically.</li>
</ol>
</div>"""


def build_html_email(report_fragment, date_str):
    """Wrap the sanitized report body in the trusted, dark-themed email shell."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#0f120d">
<style>{EMAIL_STYLE}</style>
</head>
<body style="background:#0f120d;color:#e6e4db;">
<div class="wrap" style="background:#0f120d;">
<div class="email" style="background:#0f120d;color:#e6e4db;">
<div class="masthead">
<div class="title">Weekly Open Roles <span class="accent">Scan</span></div>
<div class="date">{date_str}</div>
</div>
{VERIFY_BANNER}
{report_fragment}
<div class="footer">Generated automatically from live web search. Verify every role and source before applying.</div>
</div>
</div>
</body>
</html>"""


def verify_smtp_credentials():
    """Pre-flight: confirm the Gmail credentials authenticate, run BEFORE the
    expensive scan so an expired/revoked app password aborts the run in seconds
    instead of discarding a paid generation at the send step.

    Only a hard authentication rejection is treated as fatal — it is
    deterministic and would fail the real send too. A transient connection or
    network error here is NOT fatal: blocking an otherwise-good run on a
    momentary blip would be worse than proceeding, and the send step already has
    its own bounded retry. As everywhere in this script, never log the exception
    message or the address (the repo is public) — only the exception class name.
    """
    sender = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, app_password)
    except smtplib.SMTPAuthenticationError:
        # Deterministic credential failure — almost always an expired/revoked
        # app password. Fail now, before any web-search spend. Keep the message
        # content-free (no address, no server response) for the public log.
        raise RuntimeError(
            "Gmail rejected the credentials on a pre-flight check, before the "
            "scan ran. The GMAIL_APP_PASSWORD secret is almost certainly expired "
            "or revoked: generate a new app password at "
            "https://myaccount.google.com/apppasswords (2-Step Verification must "
            "be enabled) and update the secret. Run aborted before any spend."
        ) from None
    except (smtplib.SMTPException, OSError, TimeoutError) as exc:
        # Couldn't complete the check for a transient reason. Don't let that
        # block the run; the send step's retry covers transient send failures.
        print(
            f"Pre-flight SMTP check inconclusive ({type(exc).__name__}); "
            "proceeding with the run.",
            file=sys.stderr,
        )


def send_email(body):
    sender = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = sender
    msg["Subject"] = f"Weekly Open Roles Scan — {datetime.now(timezone.utc).strftime('%B %d, %Y')}"
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, sender, msg.as_string())


# A generation run is expensive, so a transient SMTP failure shouldn't silently
# lose it. We retry the send rather than archiving the report anywhere, because
# this repo is public and the report renders the CANDIDATE_PROFILE secret.
SEND_MAX_ATTEMPTS = 3
# Delay before each retry, indexed by the attempt that just failed. With
# SEND_MAX_ATTEMPTS == 3 only the first two rungs (30s, 120s) are reached; 300s
# is the next rung if the attempt count is ever raised.
SEND_BACKOFF_SECONDS = (30, 120, 300)


def send_with_retry(body):
    """Send the report with bounded, backed-off retries on transient failures.

    Retries on transient network/SMTP errors only. Logs the attempt number and
    the exception's class name — never the exception message (it can echo the
    recipient address or server response), the recipient address, or any part
    of the report body, because workflow logs on a public repo are public. On
    final failure raise a generic, content-free error and accept the lost run.
    """
    for attempt in range(1, SEND_MAX_ATTEMPTS + 1):
        try:
            send_email(body)
            return
        except (smtplib.SMTPException, OSError, TimeoutError) as exc:
            if attempt == SEND_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Send failed after {SEND_MAX_ATTEMPTS} attempts; "
                    "report discarded."
                ) from None
            delay = SEND_BACKOFF_SECONDS[attempt - 1]
            print(
                f"Send attempt {attempt} of {SEND_MAX_ATTEMPTS} failed "
                f"({type(exc).__name__}); retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)


if __name__ == "__main__":
    try:
        # Verify the Gmail credentials up front so an expired app password fails
        # the run in seconds rather than after a paid, search-heavy scan whose
        # report can't be persisted anywhere (public repo) and is lost if unsent.
        verify_smtp_credentials()
        report_fragment = get_newsfeed()
        date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
        newsfeed = build_html_email(report_fragment, date_str)
        # INVARIANT: this repo is public. Report CONTENT must never reach any
        # publicly readable surface — no uploaded run outputs, no workflow
        # logs, no committed report text. A paid run is protected by retrying
        # the send, not by writing the report anywhere durable.
        # Exception: get_newsfeed() writes state/seen_roles.json locally (see
        # dedupe_against_history above) — a small set of URL hashes and dates,
        # never plaintext company/title/URL/report content. The workflow YAML
        # commits that file back to the repo after a successful run. Don't add
        # any other persistence beyond that one narrow, deliberate exception.
        send_with_retry(newsfeed)
    except Exception as exc:
        # Exit non-zero so the GitHub Action surfaces the failure instead of
        # reporting a green run after a bad or missing send.
        print(f"Newsfeed run failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Sent successfully.")

