# Unified daily recommendations

The only scheduled producer is `.github/workflows/main.yml` in this repository.
PaperFlow's former independent daily producer and both periodic workflows have
no schedules. Weekly/monthly/yearly recall is disabled by user preference.

One complete, version-consistent Zotero read drives both channels. Eligible
references remain journalArticle/conferencePaper/preprint with nonblank title
and abstract, deduplicated by DOI/arXiv/title. Base weights are normalized
`1/(1+log10(rank+1))`; the final mixture is 30% base plus 70% of the normalized
base multiplied by a 14-day half-life inside the most recent 30 days.

Read all new/cross-listed records in the configured public RSS announcement.
Exclude saved and previously formally delivered papers, embed the entire raw
eligible pool using SiliconFlow BAAI/bge-m3, and form a deduplicated shortlist
of at most 100: 50 global weighted-cosine results, 40 strongest recent-reference
matches, category coverage and global fill. This prevents arbitrary RSS order
or a fixed keyword prefilter from hiding new interests. Ranking always uses
the explicit weighted mean cosine, not an LLM's numerical answer.

GLM-5.3-Flash checks candidates in descending-similarity batches of ten. Stop
after finding 30 verified matches, since lower-scored papers cannot enter the
top30. If fewer than five are verified, additional samples are clearly labelled
diagnostic and never recorded as formal recommendations. Summaries are Chinese
and generated in batches of five, with invalid/omitted peers retried separately.
No LLM-generated affiliation claims, full-paper fetches or second-model veto.
Only proposed method transfers get a compact problem/input compatibility check
by the same GLM, comparing the actually estimated variables instead of shared
words. Generic backbones or incompatible sensing assumptions cannot be used to
fill the list. Its cache is separate, so tightening this check does not force
regeneration of otherwise valid primary decisions or summaries.
The cross-domain formal-list gate is conservative: a source quote must identify
a concrete geometric/correspondence module, not merely a shared neural backbone
or training paradigm. A second model answer cannot override this local evidence
requirement. Explicit independent 3D/robotic/geometric tasks do not need to be
reinterpreted as localization in order to qualify.

Repeated reference abstracts are sent once per request. Private gate decisions
are cached under the reference digest, context, scope, model and prompt.
New/edited Zotero records invalidate gates. Public-paper summaries can be reused
across profile changes. Public vectors and authenticated encrypted checkpoints
survive retries. Model-reported tokens contribute to a public estimated-price audit.

Before sending, read Zotero again and require the version and reference digest
to match. Freeze a content-hash bundle. Both renderers consume that exact bundle
and show its first 12 digest characters. Store it encrypted with AES-256-GCM on
the state branch; the key is a GitHub Secret. No raw Zotero titles, keys, reasons
or credentials enter public artifacts.

SMTP acknowledgement and every Feishu card acknowledgement are persisted
independently. Failure in one channel does not prevent trying the other.
Retries send only missing channels/cards and reuse the frozen bundle without
another ranking or summary. If no channel has started and Zotero changed,
regenerate the unsent bundle. Once delivery starts, finish the same frozen
result. Mark complete only when both channels acknowledge the same ordered IDs.
SMTP acceptance proves submission to the mail server, not inbox arrival.

Four daily attempts remain for delivery recovery, with at most one newly
computed/delivered bundle per Beijing day; confirmed retries do not calculate
additional lists or automatically flood the user with queued historical days.
On quiet days the previously
authorized evening historical review may show five labelled samples; this is
neither a weekly/monthly/yearly scan nor a claim of new submissions. GitHub
schedules are best-effort. Local PaperFlow services are unnecessary.

Secrets: SILICONFLOW_API_KEY, ZOTERO_ID, ZOTERO_KEY, UNIFIED_STATE_KEY (base64 of
32 random bytes), SENDER, RECEIVER, SENDER_PASSWORD, FEISHU_APP_ID,
FEISHU_APP_SECRET, FEISHU_CHAT_ID. Existing SMTP credentials remain in GitHub.
