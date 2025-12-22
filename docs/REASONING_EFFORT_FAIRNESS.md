No — they’re **not all equivalent**, even though two rows *look* like they are.

### 1) “OpenAI RAG gpt-5.1” vs “CustomGPT gpt-5.1 (via API)”

These **can be equivalent** *if* they both call the **same OpenAI model snapshot** and the **same reasoning settings**.

* OpenAI’s `gpt-5.1` is an **alias**, and OpenAI lists a specific snapshot you can pin: **`gpt-5.1-2025-11-13`**. Pinning is how you make behavior consistent across runs/providers. ([OpenAI Platform][1])
* GPT-5.1 model limits & pricing (OpenAI): **400k context**, **128k max output**, **$1.25 / 1M input**, **$0.125 / 1M cached input**, **$10 / 1M output**. ([OpenAI Platform][1])

**Big fairness gotcha:** in OpenAI’s Chat Completions API reference, GPT-5.1 supports `reasoning_effort` values **`none|low|medium|high`** and the **default is `none`**. So two “gpt-5.1” runs can be *very* different if one provider sets effort and the other doesn’t. ([OpenAI Platform][2])

### 2) “Gemini RAG gemini-3-pro-preview”

This is **not equivalent** to GPT-5.1; it’s a different vendor model with different defaults, limits, and cost structure.

* Gemini 3 Pro Preview model id: **`gemini-3-pro-preview`**, with **1,048,576 max input tokens** and **65,536 max output tokens** (Vertex docs). ([Google Cloud Documentation][3])
* Gemini pricing (AI Studio / Developer API): **$2 / 1M input** and **$12 / 1M output** for prompts **≤200k tokens**, but it jumps to **$4 / 1M input** and **$18 / 1M output** for prompts **>200k tokens**. Output pricing explicitly includes “thinking tokens.” ([Google AI for Developers][4])
* Gemini 3 “thinking” control: `thinkingLevel` can be **low or high**, and if you don’t set it, Gemini 3 Pro Preview uses default dynamic thinking level **high**. Also: **you cannot disable thinking** for Gemini 3 Pro. ([Google AI for Developers][5])

### Cost + “thinking budget” comparison (what matters for your shootout)

| Row in your table                  | Underlying model                                                                         | Default “thinking”                                                                | How to set thinking budget                                                                                                           | Key pricing notes                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| OpenAI RAG: `gpt-5.1`              | OpenAI GPT-5.1                                                                           | `reasoning_effort` default **none** ([OpenAI Platform][2])                        | `reasoning_effort: none/low/medium/high` ([OpenAI Platform][2]); reasoning tokens are billed as output tokens ([OpenAI Platform][6]) | $1.25 in / $10 out per 1M tokens ([OpenAI Platform][1])                                               |
| CustomGPT: `gpt-5.1 (via API)`     | **Same as above if you pass the same model id/snapshot + params** ([OpenAI Platform][1]) | Depends on what you set in your OpenAI call ([OpenAI Platform][2])                | Same knobs as above ([OpenAI Platform][2])                                                                                           | Same OpenAI pricing if it’s truly OpenAI pass-through ([OpenAI Platform][1])                          |
| Gemini RAG: `gemini-3-pro-preview` | Google Gemini 3 Pro Preview                                                              | Default dynamic thinking **high** (can’t disable) ([Google AI for Developers][5]) | `thinkingLevel: low/high` ([Google AI for Developers][5])                                                                            | Tiered pricing at 200k prompt tokens; output includes thinking tokens ([Google AI for Developers][4]) |

### What to do to make the shootout “fair”

1. **Pin versions where possible**

* OpenAI: use **`gpt-5.1-2025-11-13`** (not just `gpt-5.1`). ([OpenAI Platform][1])
* Gemini: it’s a **preview** model id (`gemini-3-pro-preview`), so log date/time + model id for reproducibility. ([Google Cloud Documentation][3])

2. **Normalize thinking**

* If you run Gemini at its default (high), you probably want OpenAI **not** at `none`. Otherwise you’re comparing “high thinking” vs “no reasoning.” ([Google AI for Developers][5])
* Pick a policy like **High vs High** (Gemini `thinkingLevel=high`; OpenAI `reasoning_effort=high`) or **Low vs Low**.

3. **Normalize token ceilings**

* OpenAI: cap total generation (including reasoning tokens) using a max output limit; reasoning tokens are billed and can be substantial. ([OpenAI Platform][6])
* Gemini: cap `maxOutputTokens`, and remember output pricing includes thinking tokens. ([Google AI for Developers][4])

4. **Keep prompt lengths under 200k tokens** if you want Gemini costs comparable (otherwise you’ll trigger the higher pricing tier). ([Google AI for Developers][4])

### About your specific question: “what does CustomGPT `custom_model = gpt-5.1` map to?”

From OpenAI’s side, **`gpt-5.1` is a first-class OpenAI model name/alias** with an available pinned snapshot **`gpt-5.1-2025-11-13`**. ([OpenAI Platform][1])
So **if** CustomGPT is passing the string through unchanged, it maps to that OpenAI model family. The only way it becomes “unfair” is if CustomGPT (or the OpenAI RAG baseline) is silently using different **`reasoning_effort`** (or different max-token caps). ([OpenAI Platform][2])

* [reuters.com](https://www.reuters.com/technology/openai-launches-gpt-52-ai-model-with-improved-capabilities-2025-12-11/?utm_source=chatgpt.com)
* [theverge.com](https://www.theverge.com/ai-artificial-intelligence/842529/openai-gpt-5-2-new-model-chatgpt?utm_source=chatgpt.com)
* [theverge.com](https://www.theverge.com/news/802653/openai-gpt-5-1-upgrade-personality-presets?utm_source=chatgpt.com)

[1]: https://platform.openai.com/docs/models/gpt-5.1 "GPT-5.1 Model | OpenAI API"
[2]: https://platform.openai.com/docs/api-reference/chat?locale=en "Chat Completions | OpenAI API Reference"
[3]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro "Gemini 3 Pro  |  Generative AI on Vertex AI  |  Google Cloud Documentation"
[4]: https://ai.google.dev/gemini-api/docs/pricing "Gemini Developer API pricing  |  Gemini API  |  Google AI for Developers"
[5]: https://ai.google.dev/gemini-api/docs/thinking "Gemini thinking  |  Gemini API  |  Google AI for Developers"
[6]: https://platform.openai.com/docs/guides/reasoning "Reasoning models | OpenAI API"

