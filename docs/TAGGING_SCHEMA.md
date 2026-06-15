# Topic-segment tagging schema

This is the contract for `cache/topic_segments.json` — the third granularity layer
(video → topic_segment → scene → keyframe). One file, list of segment records.

## Fixed taxonomies (use these EXACT strings)

### lessons_categories (multi-label)
```
sales            hiring             leadership          marketing
operations       pricing            scaling             mindset
finance          customer_acquisition  branding         content
partnerships     personal_development  none
```
- `none` = decorative/connective content (intros, outros, B-roll, banter, transitions)
- Multi-label OK. Most segments will have 1–3.

### industries (multi-label)
```
general_business    saas             fitness            agency
ecommerce           real_estate      services           finance
healthcare          education        retail             none
```
- Most ACQ content is `general_business`. Be specific when the segment names an industry.

### audience (multi-label)
```
business_owners    employees    leadership    personal_life_advice    general_all
```
- Use what the spec gave us. Multi-label OK.

### age_group (single-label)
```
early_career_18_30    mid_career_25_45    established_35_60    general_all
```
- Single-label. Default to `general_all` when no specific cohort.

## Per-segment schema

```json
{
  "video_id":        "g3QTLRlmevc",
  "segment_idx":     0,
  "start_s":         489.5,
  "end_s":           547.2,
  "topic_title":     "Patient leadership and reading context",
  "summary":         "Sharran argues leaders need to absorb information before reacting. Leila adds the fire-triage analogy: not every problem deserves a response.",
  "lessons_summary": "Patient leadership; reading context before reacting; choosing battles; managing knee-jerk reactions",
  "expected_queries": [
    "sharran on patient leadership",
    "the part about reading context as a leader",
    "leadership and not reacting"
  ],
  "lessons_categories": ["leadership", "mindset"],
  "industries":        ["general_business"],
  "audience":          ["leadership", "business_owners"],
  "age_group":         "established_35_60"
}
```

## Writing rules (preemptive moves for searchability)

1. **Editor-language, not taxonomy-language.** `lessons_summary` should sound like how an editor talks. Pack 2–3 natural phrasings, semicolon-separated.

2. **Include names.** If Sharran is speaking, write `"Sharran on..."` not `"Speaker on..."`.

3. **Distinctive verbs.** `"Sharran reveals"`, `"Leila argues"`, `"Alex breaks down"` — adds specificity over generic `"discusses"`.

4. **Concrete topical anchors.** If a segment uses the Nokia story to explain leadership, write `"Uses Nokia story to explain..."` — the word "Nokia" becomes searchable.

5. **expected_queries = 2–3 realistic editor searches** that should hit this segment. Include synonyms editors would use (`"motivational"`, `"inspirational"`, `"pumped up"`, etc.) when relevant.

6. **Segment boundaries should be topical, not visual.** A 90-second story about Nokia is ONE segment, even if the camera cuts 8 times within it. Aim for 30s–4min segments.

7. **lessons_categories=["none"]** for purely decorative content (intros, outros, B-roll, transitions). Don't strain to find a lesson in pure connective tissue.

## Validation rules (auto-enforced)

A segment is VALID if:
- All required fields present
- `start_s` < `end_s`
- `lessons_categories` ⊆ taxonomy (no free-text categories)
- `industries` ⊆ taxonomy
- `audience` ⊆ taxonomy
- `age_group` ∈ taxonomy (or omitted, defaults to general_all)
- `expected_queries` has 2–3 entries
- `lessons_summary` is non-empty (use "none — decorative content" for decorative)
- `topic_title` < 80 chars
- `summary` < 300 chars
