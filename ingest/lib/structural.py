"""Deterministic structural-dims helper.

Stage 1 (retrieval) hard-filters out anything that doesn't match the applied
structural constraints. So by the time we're handing candidates to the judge,
every candidate has already passed the structural test. The judge doesn't need
to re-evaluate identity, format, co-presence, etc. — it just needs to know
which constraints were applied so it doesn't accidentally re-score them.

This module returns a small list of human-readable strings describing the
verified dimensions. It is the only place where filter-name <-> display-label
mapping lives.
"""
from __future__ import annotations

from typing import Optional


def compute_dims_satisfied(
    speaker: Optional[str] = None,
    required_speakers: Optional[list[str]] = None,
    is_animation: Optional[bool] = None,
    talking_head_pose: Optional[str] = None,
    speakers_count: Optional[str] = None,
    max_age_days: Optional[int] = None,
    min_age_days: Optional[int] = None,
    visual_concept: Optional[str] = None,
) -> list[str]:
    """Return a flat list of short labels describing the verified structural dims.

    Examples:
      [] when no filters were applied
      ["speaker=alex (voice match verified)"]
      ["required_speakers=[leila, sharran] verified by audio diarization"]
      ["format=whiteboard (visual classifier verified)"]
    """
    out: list[str] = []
    if required_speakers:
        out.append(
            f"required_speakers=[{', '.join(required_speakers)}] verified by audio diarization"
        )
    elif speaker:
        out.append(f"speaker={speaker} (audio voice match verified)")
    if is_animation is True:
        out.append("is_animation=true (visual classifier verified)")
    elif is_animation is False:
        out.append("is_animation=false (live-footage frame, visual classifier verified)")
    if talking_head_pose == "front_view":
        out.append("talking_head_pose=front_view (visual classifier verified)")
    elif talking_head_pose == "none":
        out.append("talking_head_pose=none (no foreground talking head)")
    if speakers_count:
        out.append(f"speakers_count={speakers_count} (audio diarization verified)")
    if max_age_days is not None:
        if max_age_days <= 7:
            out.append(f"uploaded in the last {max_age_days} days (verified)")
        elif max_age_days <= 30:
            out.append("uploaded within the last month (verified)")
        elif max_age_days <= 90:
            out.append(f"uploaded within the last {max_age_days} days (verified)")
        else:
            out.append(f"uploaded in the last {max_age_days // 30} months (verified)")
    if visual_concept:
        out.append(f'visual: "{visual_concept}" (CLIP visual retrieval matched this)')
    if min_age_days is not None:
        if min_age_days >= 365:
            out.append(f"uploaded over {min_age_days // 365}+ years ago (verified)")
        elif min_age_days >= 30:
            out.append(f"uploaded over {min_age_days // 30} month(s) ago (verified)")
        else:
            out.append(f"uploaded over {min_age_days} days ago (verified)")
    return out


def compose_why(structural_dims: list[str], judge_reason: str) -> str:
    """Produce the display 'why' string from structural + judge.

    Keep concise: at most ~120 chars total. Prefer judge reason if structural is empty,
    otherwise lead with the strongest structural dim then the judge's topic note.
    """
    if not structural_dims and not judge_reason:
        return ""
    if not structural_dims:
        return judge_reason
    # Combine the most informative dim + judge reason (which may add topical color)
    lead = structural_dims[0]
    if judge_reason:
        return f"{lead}. {judge_reason}"
    return lead
