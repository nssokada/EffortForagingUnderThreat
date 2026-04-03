"""
LIMA Preprocessing Pipeline - Stage 5: Participant Filtering & Harmonization
============================================================================
Purpose
-------
This stage finalizes the *analysis cohort* across multiple outputs by:

1) Loading:
   - behavior (trial-level or analysis-ready behavioral table)
   - behavior_rich (richer trial-level table; used for outcome-based filtering)
   - subjective_reports (saved as "feelings")
   - mental_health (saved as "psych")

2) Outcome-based participant filtering (escape-rate):
   - outcome == 0  → escaped
   - outcome == 1  → captured
   Participants must escape on at least `config.min_escape_rate` of trials
   (default fallback is 0.35 if not present in config).

3) Modality completeness enforcement:
   After the success/outcome filter, enforce that *every kept participant* exists in
   all required dataframes (behavior, behavior_rich, subjective_reports, mental_health).

4) Audit trail:
   Produce a human-readable report of all excluded participants and the reason(s)
   they were excluded:
   - failed_escape_rate
   - missing_subjective_report
   - missing_mental_health
   - missing_behavior
   - missing_behavior_rich
   - no_outcome_data

5) Harmonized participant mapping:
   Create a stable integer subject id `subj` shared across all kept dataframes.

Outputs
-------
- behavior.csv/.pkl
- behavior_rich.csv/.pkl   (also exposed as stage5_outputs['filtered'] for Stage 6)
- psych.csv/.pkl           (mental health)
- feelings.csv/.pkl        (subjective reports)
- subject_mapping.csv      (participantID → subj)
- participant_filter_report.txt
- participant_qc.csv       (optional but useful)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
import pandas as pd

from config import PipelineConfig
from utils import PipelineLogger, load_pickle, save_outputs, create_processing_report


def _safe_unique_ids(df: pd.DataFrame, id_col: str) -> Set[str]:
    if df is None or df.empty or id_col not in df.columns:
        return set()
    return set(df[id_col].astype(str).unique())


def _compute_escape_rate(
    behavior_rich: pd.DataFrame,
    id_col: str,
    outcome_col: str,
    logger: Optional[PipelineLogger] = None
) -> pd.Series:
    """
    Returns a Series indexed by participantID with escape-rate in [0,1].
    escape-rate = mean(outcome == 0) over non-missing outcomes.
    """
    if outcome_col not in behavior_rich.columns:
        raise KeyError(f"Missing required column '{outcome_col}' in behavior_rich")

    tmp = behavior_rich[[id_col, outcome_col]].copy()
    tmp[id_col] = tmp[id_col].astype(str)

    # Drop NA outcomes for rate computation
    tmp = tmp.dropna(subset=[outcome_col])

    if tmp.empty:
        if logger:
            logger.log("No non-missing outcome data found in behavior_rich.", "warning")
        return pd.Series(dtype=float)

    escape_rate = tmp.groupby(id_col)[outcome_col].apply(lambda s: (s == 0).mean())
    return escape_rate


def _write_filter_report(
    path: Path,
    reasons: Dict[str, List[str]],
    logger: Optional[PipelineLogger] = None
) -> None:
    """
    Write a readable report:
    - counts by reason
    - participant list per reason
    """
    lines: List[str] = []
    lines.append("PARTICIPANT FILTER REPORT (Stage 5)")
    lines.append("=" * 60)

    total_unique = len(set().union(*[set(v) for v in reasons.values()])) if reasons else 0
    lines.append(f"Total excluded participants: {total_unique}")
    lines.append("")

    # Counts
    lines.append("Counts by reason:")
    for reason, pids in sorted(reasons.items(), key=lambda x: x[0]):
        lines.append(f"- {reason}: {len(pids)}")
    lines.append("")

    # Lists
    for reason, pids in sorted(reasons.items(), key=lambda x: x[0]):
        lines.append(f"{reason} ({len(pids)}):")
        for pid in sorted(set(pids)):
            lines.append(f"  - {pid}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

    if logger:
        logger.log(f"Saved participant filter report: {path}")


def run_stage5(
    behavior_path: Path,
    behavior_rich_path: Path,
    subjective_reports_path: Path,
    mental_health_path: Path,
    output_dir: Path,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Run Stage 5: Filter and harmonize participants across datasets.

    Parameters
    ----------
    behavior_path : Path
        Pickle path to behavior dataframe (from Stage 2 or later).
    behavior_rich_path : Path
        Pickle path to rich behavior dataframe (used for outcome-based filter).
    subjective_reports_path : Path
        Pickle path to processed subjective reports (Stage 3).
    mental_health_path : Path
        Pickle path to mental health scores/items (Stage 4). Saved as "psych".
    output_dir : Path
        Directory to write outputs.
    config : PipelineConfig | None
        Pipeline configuration.

    Returns
    -------
    Dict[str, Any]
        Outputs dictionary mirroring other stages: each saved artifact has csv/pickle paths.
        Also includes 'filtered' key used by Stage 6 (points to behavior_rich output).
    """
    config = config or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger("Stage5_Filtering", output_dir)
    logger.log("Starting Stage 5: Participant filtering & harmonization")

    id_col = getattr(config, "participant_id_col", "participantID")
    outcome_col = getattr(config, "outcome_col", "outcome")
    min_escape_rate = getattr(config, "min_escape_rate", 0.35)

    # -----------------------------
    # Load inputs
    # -----------------------------
    behavior_df = load_pickle(behavior_path)
    behavior_rich_df = load_pickle(behavior_rich_path)
    subjective_df = load_pickle(subjective_reports_path)
    mental_health_df = load_pickle(mental_health_path)

    # Standardize ID columns to str
    for df_name, df in [
        ("behavior", behavior_df),
        ("behavior_rich", behavior_rich_df),
        ("subjective_reports", subjective_df),
        ("mental_health", mental_health_df),
    ]:
        if df is not None and not df.empty and id_col in df.columns:
            df[id_col] = df[id_col].astype(str)
        else:
            logger.log(f"Warning: '{df_name}' missing '{id_col}' or is empty.", "warning")

    # -----------------------------
    # Participant universes
    # -----------------------------
    p_behavior = _safe_unique_ids(behavior_df, id_col)
    p_behavior_rich = _safe_unique_ids(behavior_rich_df, id_col)
    p_subjective = _safe_unique_ids(subjective_df, id_col)
    p_mental = _safe_unique_ids(mental_health_df, id_col)

    universe = set().union(p_behavior, p_behavior_rich, p_subjective, p_mental)
    logger.log(f"Participant universe (union across inputs): {len(universe)}")

    # -----------------------------
    # 1) Escape-rate filter (based on behavior_rich)
    # -----------------------------
    reasons: Dict[str, List[str]] = {
        "failed_escape_rate": [],
        "no_outcome_data": [],
        "missing_subjective_report": [],
        "missing_mental_health": [],
        "missing_behavior": [],
        "missing_behavior_rich": [],
    }

    escape_rate = pd.Series(dtype=float)
    if behavior_rich_df is None or behavior_rich_df.empty or id_col not in behavior_rich_df.columns:
        # If we cannot compute escape rate, exclude everyone present in other modalities because we can't validate
        reasons["no_outcome_data"] = sorted(list(universe))
        keepers_after_escape: Set[str] = set()
        logger.log("Cannot compute escape rate: behavior_rich is missing/empty. Excluding all.", "error")
    else:
        try:
            escape_rate = _compute_escape_rate(behavior_rich_df, id_col=id_col, outcome_col=outcome_col, logger=logger)
        except KeyError as e:
            reasons["no_outcome_data"] = sorted(list(p_behavior_rich))
            keepers_after_escape = set()
            logger.log(str(e), "error")
        else:
            # Participants with no computed escape rate (e.g., all outcomes missing) => exclude
            computed_pids = set(escape_rate.index.astype(str))
            no_outcome = sorted(list(p_behavior_rich - computed_pids))
            reasons["no_outcome_data"].extend(no_outcome)

            failed = escape_rate[escape_rate < float(min_escape_rate)].index.astype(str).tolist()
            reasons["failed_escape_rate"].extend(failed)

            keepers_after_escape = set(p_behavior_rich) - set(failed) - set(no_outcome)

            logger.log(
                f"Escape-rate filter (min={min_escape_rate}): "
                f"keepers={len(keepers_after_escape)}; failed={len(failed)}; no_outcome={len(no_outcome)}"
            )

    # -----------------------------
    # 2) Enforce presence in all required dataframes
    #    (after escape filter)
    # -----------------------------
    missing_subj = sorted(list(keepers_after_escape - p_subjective))
    missing_mh = sorted(list(keepers_after_escape - p_mental))
    missing_beh = sorted(list(keepers_after_escape - p_behavior))
    missing_beh_rich = sorted(list(keepers_after_escape - p_behavior_rich))

    reasons["missing_subjective_report"].extend(missing_subj)
    reasons["missing_mental_health"].extend(missing_mh)
    reasons["missing_behavior"].extend(missing_beh)
    reasons["missing_behavior_rich"].extend(missing_beh_rich)

    final_keepers = set(keepers_after_escape)
    final_keepers -= set(missing_subj)
    final_keepers -= set(missing_mh)
    final_keepers -= set(missing_beh)
    final_keepers -= set(missing_beh_rich)

    logger.log(f"Final keepers after modality completeness: {len(final_keepers)}")

    # -----------------------------
    # Build QC table (one row per participant in union)
    # -----------------------------
    qc = pd.DataFrame({id_col: sorted(universe)})
    qc["kept"] = qc[id_col].isin(final_keepers)
    qc["present_in_behavior"] = qc[id_col].isin(p_behavior)
    qc["present_in_behavior_rich"] = qc[id_col].isin(p_behavior_rich)
    qc["present_in_subjective_reports"] = qc[id_col].isin(p_subjective)
    qc["present_in_mental_health"] = qc[id_col].isin(p_mental)

    # Add escape rate (NaN if not available)
    if not escape_rate.empty:
        qc = qc.merge(
            escape_rate.rename("escape_rate").reset_index().rename(columns={id_col: "index"}),
            how="left",
            left_on=id_col,
            right_on="index"
        ).drop(columns=["index"])
    else:
        qc["escape_rate"] = pd.NA

    qc_path = output_dir / "participant_qc.csv"
    qc.to_csv(qc_path, index=False)
    logger.log(f"Saved participant QC table: {qc_path}")

    # -----------------------------
    # Participant mapping: participantID -> subj
    # -----------------------------
    keepers_sorted = sorted(final_keepers)
    mapping_df = pd.DataFrame({
        id_col: keepers_sorted,
        "subj": list(range(1, len(keepers_sorted) + 1))
    })

    mapping = dict(zip(mapping_df[id_col].tolist(), mapping_df["subj"].tolist()))

    # Helper to add subj
    def _add_subj(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or id_col not in df.columns:
            return df
        out = df.copy()
        out["subj"] = out[id_col].map(mapping)
        return out

    # -----------------------------
    # Filter and add subj to all dfs
    # -----------------------------
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or id_col not in df.columns:
            return pd.DataFrame()
        return df[df[id_col].isin(final_keepers)].copy()

    behavior_kept = _add_subj(_filter(behavior_df))
    behavior_rich_kept = _add_subj(_filter(behavior_rich_df))
    feelings_kept = _add_subj(_filter(subjective_df))
    psych_kept = _add_subj(_filter(mental_health_df))

    # -----------------------------
    # Save outputs
    # -----------------------------
    outputs: Dict[str, Any] = {}

    outputs["behavior"] = save_outputs(behavior_kept, output_dir, "behavior", logger=logger)
    outputs["behavior_rich"] = save_outputs(behavior_rich_kept, output_dir, "behavior_rich", logger=logger)
    outputs["psych"] = save_outputs(psych_kept, output_dir, "psych", logger=logger)
    outputs["feelings"] = save_outputs(feelings_kept, output_dir, "feelings", logger=logger)

    outputs["subject_mapping"] = save_outputs(mapping_df, output_dir, "subject_mapping", logger=logger)

    # -----------------------------
    # Merge demographics (if available)
    # -----------------------------
    demographics_path = getattr(config, "demographics_path", None)
    if demographics_path is None:
        # Auto-detect: look for *demographics*.csv in the data root
        data_root = Path(config.raw_data_dir).parent if config else None
        if data_root:
            demo_candidates = list(data_root.glob("*demographics*.csv"))
            if demo_candidates:
                demographics_path = demo_candidates[0]

    if demographics_path and Path(demographics_path).exists():
        demo_df = pd.read_csv(demographics_path)
        if id_col in demo_df.columns:
            # Keep only key demographic columns
            demo_cols = [id_col]
            for col in ["Age", "Sex", "age", "sex", "gender", "Gender"]:
                if col in demo_df.columns and col not in demo_cols:
                    demo_cols.append(col)
            demo_kept = demo_df[demo_cols].copy()
            demo_kept = demo_kept[demo_kept[id_col].isin(final_keepers)]
            demo_kept["subj"] = demo_kept[id_col].map(mapping)
            outputs["demographics"] = save_outputs(demo_kept, output_dir, "demographics", logger=logger)
            logger.log(f"Merged demographics: {len(demo_kept)} participants ({demo_cols})")
        else:
            logger.log(f"Demographics file found but missing '{id_col}' column — skipping", "warning")
    else:
        logger.log("No demographics file found — skipping")

    report_path = output_dir / "participant_filter_report.txt"
    _write_filter_report(report_path, reasons, logger=logger)
    outputs["participant_filter_report"] = {"txt": report_path}

    outputs["participant_qc"] = {"csv": qc_path}

    # Maintain backward compatibility for Stage 6
    outputs["filtered"] = outputs["behavior_rich"]

    # Processing report (summary)
    input_info = {
        "behavior_path": str(behavior_path),
        "behavior_rich_path": str(behavior_rich_path),
        "subjective_reports_path": str(subjective_reports_path),
        "mental_health_path": str(mental_health_path),
        "min_escape_rate": float(min_escape_rate),
        "outcome_col": outcome_col,
    }
    output_info = {
        "behavior_rows": int(len(behavior_kept)),
        "behavior_rich_rows": int(len(behavior_rich_kept)),
        "feelings_rows": int(len(feelings_kept)),
        "psych_rows": int(len(psych_kept)),
        "n_keepers": int(len(final_keepers)),
        "n_excluded_total": int(len(universe - final_keepers)),
    }
    processing_stats = {
        "excluded_failed_escape_rate": len(set(reasons["failed_escape_rate"])),
        "excluded_no_outcome_data": len(set(reasons["no_outcome_data"])),
        "excluded_missing_subjective_report": len(set(reasons["missing_subjective_report"])),
        "excluded_missing_mental_health": len(set(reasons["missing_mental_health"])),
        "excluded_missing_behavior": len(set(reasons["missing_behavior"])),
        "excluded_missing_behavior_rich": len(set(reasons["missing_behavior_rich"])),
    }

    _ = create_processing_report(
        output_dir=output_dir,
        stage_name="Stage 5: Participant Filtering & Harmonization",
        input_info=input_info,
        output_info={k: {kk: str(vv) for kk, vv in v.items()} if isinstance(v, dict) else str(v) for k, v in outputs.items()},
        processing_stats=processing_stats,
        logger=logger,
    )

    logger.log("Stage 5 complete.")
    logger.save_log()

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage 5: Participant Filtering & Harmonization")
    parser.add_argument("--behavior", required=True, help="Path to behavior pickle")
    parser.add_argument("--behavior-rich", required=True, help="Path to behavior_rich pickle")
    parser.add_argument("--subjective-reports", required=True, help="Path to subjective reports pickle")
    parser.add_argument("--mental-health", required=True, help="Path to mental health pickle")
    parser.add_argument("--output", "-o", required=True, help="Output directory")

    args = parser.parse_args()

    cfg = PipelineConfig()
    run_stage5(
        behavior_path=Path(args.behavior),
        behavior_rich_path=Path(args.behavior_rich),
        subjective_reports_path=Path(args.subjective_reports),
        mental_health_path=Path(args.mental_health),
        output_dir=Path(args.output),
        config=cfg,
    )
