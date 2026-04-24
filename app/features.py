"""
Feature extraction for radiology prior-relevance classification.

Given a (current_study, prior_study) pair, produce a feature vector that
a lightweight classifier can score. Features are designed to be robust
to the quirky capitalization and abbreviations seen in study_description
fields (e.g. "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
"CT HEAD WITHOUT CNTRST").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------

# Imaging modalities we want to detect. Order matters for tie-breaking:
# more specific tokens first.
MODALITIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MR",      ("MRI", "MR ", "MAGNETIC RESONANCE", "MRA", "MRV")),
    ("CT",      ("CT ", "CAT SCAN", "COMPUTED TOMOGRAPHY", "CTA", "CTV")),
    ("XR",      ("XR", "X-RAY", "XRAY", "RADIOGRAPH", " RAD ")),
    ("US",      ("US ", "ULTRASOUND", "SONO", "DOPPLER")),
    ("ECHO",    ("ECHO ", "ECHOCARDIO", "TRANSESOPHAGEAL", "TEE ", "TTE ")),
    ("MAMMO",   ("MAMMO", "MAM ", "MAM/", "BREAST TOMO", "TOMOSYNTHESIS", "DIGITAL SCREENER", "BI RADS", "BIRADS")),
    ("NM",      ("NM ", "NUCLEAR", "PET", "SPECT", "BONE SCAN")),
    ("FLUORO",  ("FLUORO", "FLUOROSCOPY", "UPPER GI", "BARIUM")),
    ("DEXA",    ("DEXA", "BONE DENSITOMETRY", "BMD")),
)

# Anatomical regions. Radiologists care about body region overlap far
# more than modality alone — a prior CT head is useful when reading an
# MRI head, but a prior MRI knee is not useful for an MRI brain.
REGIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HEAD",        ("BRAIN", "HEAD", "SKULL", "CRANIAL", "INTRACRANIAL", "CEREBRAL")),
    ("NECK",        ("NECK", "CERVICAL", "THYROID", "LARYNX", "CAROTID")),
    ("CHEST",       ("CHEST", "THORAX", "THORACIC", "LUNG", "PULMONARY", "CARDIAC", "HEART", "ECHO", "RIBS", "RIB ", "PLEURAL", "V/Q", "VQ SCAN", "MEDIASTIN")),
    ("ABDOMEN",     ("ABDOMEN", "ABDOMINAL", "LIVER", "HEPATIC", "PANCREAS", "KIDNEY", "RENAL", "BILIARY")),
    ("PELVIS",      ("PELVIS", "PELVIC", "BLADDER", "PROSTATE", "UTERUS", "OVARY")),
    ("SPINE_L",     ("LUMBAR", "L-SPINE", "LSPINE", " L SPINE")),
    ("SPINE_T",     ("THORACIC SPINE", "T-SPINE", "TSPINE")),
    ("SPINE_C",     ("C-SPINE", "CSPINE", "CERVICAL SPINE")),
    ("SPINE",       ("SPINE", "VERTEBRA")),
    ("SHOULDER",    ("SHOULDER", "CLAVICLE", "SCAPULA")),
    ("ELBOW",       ("ELBOW",)),
    ("WRIST_HAND",  ("WRIST", "HAND", "FINGER")),
    ("HIP",         ("HIP", "FEMUR PROX")),
    ("KNEE",        ("KNEE", "PATELLA")),
    ("ANKLE_FOOT",  ("ANKLE", "FOOT", "TOE", "CALCANEUS")),
    ("FEMUR",       ("FEMUR",)),
    ("TIBIA",       ("TIBIA", "FIBULA")),
    ("HUMERUS",     ("HUMERUS",)),
    ("BREAST",      ("BREAST", "MAMMOGRAM", "MAMMOGRAPHY", "MAMMO", "MAM ", "MAM/", "DIGITAL SCREENER", "TOMO", "BI RADS", "BIRADS")),
    ("VASCULAR",    ("AORTA", "VENOGRAM", "ARTERIOGRAM", "EXTREMITY RUNOFF")),
)

# Clinical context hints. Same modality + same region + same context is
# about the strongest possible relevance signal.
CONTEXT_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("STROKE",      ("STROKE", "INFARCT", "TIA")),
    ("TRAUMA",      ("TRAUMA", "FRACTURE", "FX ")),
    ("ONCOLOGY",    ("TUMOR", "MASS", "CANCER", "MALIG", "MET ", "METS", "ONC ", "ONCOLOGY")),
    ("CONTRAST",    ("WITH CONTRAST", "W CONTRAST", "W/ CONTRAST", "W/CONTRAST", "WCONTRAST")),
    ("NOCONTRAST",  ("WITHOUT CONTRAST", "WO CONTRAST", "W/O CONTRAST", "WOCONTRAST", "WITHOUT CNTRST", "NO CONTRAST")),
    ("MSK",         ("ARTHRITIS", "JOINT", "EFFUSION")),
    ("VASCULAR_C",  ("ANEURYSM", "DISSECTION", "EMBOLISM", "PE ")),
)

# Whitespace / punctuation normalization
_NORM_RE = re.compile(r"[^A-Z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Uppercase, strip punctuation, collapse whitespace, pad with spaces.

    Padding with leading/trailing spaces lets us match tokens like "CT "
    or " L SPINE" without false positives on substrings.
    """
    if not text:
        return " "
    t = text.upper()
    t = _NORM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return f" {t} "


def _match_any(text: str, tokens: Iterable[str]) -> bool:
    for tok in tokens:
        if tok in text:
            return True
    return False


def extract_modality(text: str) -> str:
    t = normalize(text)
    for label, tokens in MODALITIES:
        if _match_any(t, tokens):
            return label
    return "UNK"


def extract_regions(text: str) -> set[str]:
    """A study can touch multiple regions (e.g. 'CT CHEST ABDOMEN PELVIS')."""
    t = normalize(text)
    regions: set[str] = set()
    for label, tokens in REGIONS:
        if _match_any(t, tokens):
            regions.add(label)
    # Collapse specific spine labels into the generic SPINE bucket too,
    # so a prior "MRI SPINE" still overlaps with "MRI LUMBAR SPINE".
    if regions & {"SPINE_L", "SPINE_T", "SPINE_C"}:
        regions.add("SPINE")
    return regions


def extract_laterality(text: str) -> str:
    """Return 'L', 'R', 'B' (bilateral), or 'U' (unknown)."""
    t = normalize(text)
    # Check bilateral first — 'BILAT' appearing means both sides.
    if ' BILAT' in t or ' BILATERAL' in t or ' BI ' in t or ' BOTH ' in t:
        return 'B'
    # Left/right with word boundaries via padding
    has_l = ' LT ' in t or ' LEFT ' in t or ' L ' in t
    has_r = ' RT ' in t or ' RIGHT ' in t or ' R ' in t
    if has_l and not has_r:
        return 'L'
    if has_r and not has_l:
        return 'R'
    return 'U'


def extract_contexts(text: str) -> set[str]:
    t = normalize(text)
    ctx: set[str] = set()
    for label, tokens in CONTEXT_TOKENS:
        if _match_any(t, tokens):
            ctx.add(label)
    return ctx


# -----------------------------------------------------------------------------
# Date handling
# -----------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    # Dataset uses YYYY-MM-DD but be tolerant.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


def days_between(current: str | None, prior: str | None) -> int | None:
    c = _parse_date(current)
    p = _parse_date(prior)
    if c is None or p is None:
        return None
    return (c - p).days


# -----------------------------------------------------------------------------
# Feature vector
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PairFeatures:
    exact_desc_match: int           # 1 if normalized descriptions are identical
    same_modality: int              # 1 if modalities match and are known
    modality_both_unknown: int      # 1 if both failed modality detection
    region_overlap: int             # count of overlapping regions
    region_jaccard: float           # |intersect| / |union|, 0 if both empty
    any_region_match: int           # 1 if region_overlap >= 1
    context_overlap: int            # count of overlapping clinical contexts
    contrast_conflict: int          # 1 if one says WITH and other WITHOUT
    days_delta: int                 # absolute days; very large => likely stale
    is_recent: int                  # 1 if |delta| <= 365
    is_very_recent: int             # 1 if |delta| <= 90
    is_stale: int                   # 1 if |delta| > 365 * 5
    missing_date: int               # 1 if either date unparseable
    prior_desc_len: int             # crude complexity signal
    laterality_conflict: int        # 1 if one side L and other R (never both in same body part)
    laterality_match: int           # 1 if both sides identified and equal

    def to_vector(self) -> list[float]:
        return [
            float(self.exact_desc_match),
            float(self.same_modality),
            float(self.modality_both_unknown),
            float(self.region_overlap),
            float(self.region_jaccard),
            float(self.any_region_match),
            float(self.context_overlap),
            float(self.contrast_conflict),
            float(self.days_delta) / 365.0,   # scale to years
            float(self.is_recent),
            float(self.is_very_recent),
            float(self.is_stale),
            float(self.missing_date),
            float(self.prior_desc_len) / 50.0,
            float(self.laterality_conflict),
            float(self.laterality_match),
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "exact_desc_match",
            "same_modality",
            "modality_both_unknown",
            "region_overlap",
            "region_jaccard",
            "any_region_match",
            "context_overlap",
            "contrast_conflict",
            "years_delta",
            "is_recent_1y",
            "is_very_recent_90d",
            "is_stale_5y",
            "missing_date",
            "prior_desc_len_scaled",
            "laterality_conflict",
            "laterality_match",
        ]


def extract_pair_features(
    current_desc: str,
    current_date: str | None,
    prior_desc: str,
    prior_date: str | None,
) -> PairFeatures:
    cur_norm = normalize(current_desc)
    pri_norm = normalize(prior_desc)

    cur_mod = extract_modality(current_desc)
    pri_mod = extract_modality(prior_desc)
    cur_regs = extract_regions(current_desc)
    pri_regs = extract_regions(prior_desc)
    cur_ctx = extract_contexts(current_desc)
    pri_ctx = extract_contexts(prior_desc)

    union = cur_regs | pri_regs
    inter = cur_regs & pri_regs
    jaccard = (len(inter) / len(union)) if union else 0.0

    contrast_conflict = int(
        ("CONTRAST" in cur_ctx and "NOCONTRAST" in pri_ctx)
        or ("NOCONTRAST" in cur_ctx and "CONTRAST" in pri_ctx)
    )

    cur_lat = extract_laterality(current_desc)
    pri_lat = extract_laterality(prior_desc)
    lat_conflict = int(
        (cur_lat == 'L' and pri_lat == 'R') or (cur_lat == 'R' and pri_lat == 'L')
    )
    lat_match = int(cur_lat in ('L', 'R', 'B') and cur_lat == pri_lat)

    delta = days_between(current_date, prior_date)
    missing = 1 if delta is None else 0
    abs_delta = abs(delta) if delta is not None else 0
    # Gate recency flags on having a real date — otherwise a missing
    # date would incorrectly register as 'very recent' (0 days old).
    has_date = 0 if missing else 1

    return PairFeatures(
        exact_desc_match=int(cur_norm == pri_norm),
        same_modality=int(cur_mod == pri_mod and cur_mod != "UNK"),
        modality_both_unknown=int(cur_mod == "UNK" and pri_mod == "UNK"),
        region_overlap=len(inter),
        region_jaccard=jaccard,
        any_region_match=int(len(inter) >= 1),
        context_overlap=len(cur_ctx & pri_ctx),
        contrast_conflict=contrast_conflict,
        days_delta=abs_delta,
        is_recent=int(has_date and abs_delta <= 365),
        is_very_recent=int(has_date and abs_delta <= 90),
        is_stale=int(has_date and abs_delta > 365 * 5),
        missing_date=missing,
        prior_desc_len=len(prior_desc or ""),
        laterality_conflict=lat_conflict,
        laterality_match=lat_match,
    )
