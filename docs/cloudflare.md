# Cloudflare Workers AI configuration

Both daily mail and periodic recall use these Actions Secrets:
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Never commit their values.
The base URL is
`https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1`.

The `CUSTOM_CONFIG` repository variable must select
`llm.generation_kwargs.model: '@cf/zai-org/glm-4.7-flash'` and
`reranker.api.model: '@cf/baai/bge-m3'` with `batch_size: 32`.
Keep the API key/base URL fields as environment-variable references; the
workflow supplies `OPENAI_API_KEY`, `OPENAI_API_BASE` and `OPENAI_BASE_URL`.
`config/custom.yaml` provides the matching example.

The workflow sets `PAPERFLOW_RELEVANCE_REVIEW_MODEL` to
`@cf/openai/gpt-oss-120b`. GLM first screens task/method relevance and creates
Chinese TLDRs; the independent reviewer can veto accepted decisions. Numerical
ranking remains BGE-M3 cosine similarity with the existing 70% recent / 30%
stable corpus weighting. Maximums remain 100 candidates and 30 outputs.

GLM thinking is disabled using `chat_template_kwargs.enable_thinking=false`;
the independent reviewer uses low reasoning effort. Logs record model token
and neuron usage. Review prompts keep repeated reference abstracts once in a
keyed catalog while preserving every supplied reference. Invalid or omitted
per-paper decisions stay unverified; they cannot become formal recommendations
or discard valid peers. Malformed responses, ambiguous IDs and API failures
still fail the run without claiming delivery.

All three models and both projects share the account's daily
free allocation. There is no paid provider fallback, but Cloudflare account
billing settings still apply. Keep a free plan if relying on quota exhaustion
to stop requests rather than incur charges.

An acknowledged SMTP receipt completes a queued announcement batch, not a
green workflow. A receipt proves SMTP acceptance, not inbox placement. Retry
without `force_send` to avoid duplicating completed mail. Automatic periodic
recall waits while daily batches are pending, then resumes the older periods.
Free quota may require a backlog to be recovered over several days.
