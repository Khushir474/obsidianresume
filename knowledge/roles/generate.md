# Role File Generator

**How to use:** Tell Claude — "Read `roles/generate.md` and generate a new role file for me."
Claude will ask you questions, then write a complete `roles/[role_name].md` file ready to use.

---

## Generator Instructions (Claude reads and follows these)

You are generating a new role-specific ATS prompt file for the Resume Builder skill.
Ask the user the questions below using **AskUserQuestion**, then write the complete role file.

---

### Question 1 — Role identity
Ask:
> "What role are you targeting?"

Collect:
- Role name (e.g. "DevOps Engineer", "Marketing Analyst", "Cybersecurity Analyst")
- Industry or domain context if relevant (e.g. "fintech", "healthcare", "e-commerce")

---

### Question 2 — Core emphasis
Ask:
> "What is the core emphasis of this role — what does success look like on day 90?"

Use the answer to write the EXTRAPOLATION section. The ownership arc for this role type
should reflect what a strong practitioner would have done, not just what was written.

---

### Question 3 — Stack and tools
Ask:
> "What tools, platforms, and languages does this role require? List as many as you know —
> include the ones the market expects even if they're not always in JDs."

These become the foundation of List A and the ROLE-SPECIFIC ATS SIGNALS stack line.
If the user is unsure, fill in the most widely adopted tools for this role in the 2026 market.

---

### Question 4 — Keyword lists
Ask:
> "What are the key soft skills, business outcomes, and process terms this role cares about?
> (e.g. stakeholder comms, delivery velocity, cost savings, compliance)"

These populate List B and List C.

---

### Question 5 — Writing register
Ask:
> "Should bullets lead with business outcomes or technical methods for this role?
> And is there any language to suppress (e.g. analyst-sounding words for engineering roles)?"

Use this to write the WRITING STYLE section. If the role is engineering-focused, add a
SUPPRESS THIS LANGUAGE block (see `ai_ml_engineer.md` as reference).
If the role is process/ops-focused, add a PROCESS LANGUAGE RULE block (see `product_analyst_ops.md`).

---

### Output

After collecting answers, write a complete role file following this exact structure
(do not add sections or skip sections):

```
STEP 1 — BEFORE WRITING ANYTHING:
Extract [two or three] lists from the JD:
A) Hard keywords: every tool, language, platform, and methodology named
B) Soft keywords: every trait or competency named
[C) only add if role has distinct process/domain terms worth separating]
All items from all lists must appear in the resume.
Show me Lists A[, B, and C] before writing.

STEP 2 — WRITE THE RESUME. Follow every rule below:

ATS RULES (non-negotiable):
[standard rules + any role-specific keyword weighting, e.g. "SQL must appear in..."]

ROLE-SPECIFIC ATS SIGNALS TO HIT:
[5 compact lines: one per signal category, comma-separated terms]

[SUPPRESS THIS LANGUAGE block — only for engineering roles]
[PROCESS LANGUAGE RULE block — only for ops/coordination roles]

BULLET STRUCTURE RULE:
Every bullet must contain BOTH the technical method AND the business
outcome — not one or the other. ATS needs the method for keyword
scoring; humans need the outcome for relevance.

Right: "[example tailored to this role type]"

Wrong (ATS fails): "[example]" — [tool] invisible, keyword score drops
Wrong (human fails): "[example]" — no business context, recruiter skips it

EXTRAPOLATION:
Fill the gap between what I wrote and what a strong [role] would
have done. [3–4 specific if/then scenarios for this role type.]
Push to what I can defend in an interview.

WRITING STYLE — Paul Graham (human layer, applied after ATS):
[4–5 bullets: verbs to use, framing approach, what to avoid]

FORMAT:
Technical Skills — grouped by: Languages | [categories relevant to role] | Tools
                   All List A keywords present, full names + abbreviations both included
Work Experience  — 3-4 bullets per role, method + outcome in each,
                   dates as "Month YYYY – Month YYYY"
Education        — Degree, Major, Institution, [coursework note if applicable], Graduation Year

At the bottom, add:
"EXTRAPOLATED — PREP BEFORE INTERVIEW"
For each addition: what was added | what it assumes | one sentence
on how to talk about it honestly
```

Save the output to: `roles/[role_name_lowercase_underscored].md`
Then tell the user:
1. The file path where it was saved
2. To update the relevance map in `skill.md` to include this role's experience file preferences
3. Any signals they should verify are accurate for their specific target market
