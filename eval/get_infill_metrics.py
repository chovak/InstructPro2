#!/usr/bin/env python3
"""
Protein Infilling Recovery Metrics
-----------------------------------
Computes per-mask sequence recovery (BLOSUM62 global alignment identity)
between ground-truth infill spans (target.txt) and model-generated infill
spans (design.txt), across one or more checkpoint output directories.

Each directory is expected to contain:
    design.txt   <id>\t<raw generated sequence with <M#>/<SEP> tags>
    target.txt   <id>\t<raw ground-truth sequence with <M#>/<SEP> tags>

Infilling examples are identified by the presence of <SEP> in the TARGET
line. Lines without <SEP> in the target are treated as autoregressive and
skipped for this metric.

POSTPROCESSING OPTION
----------------------
Set run_both_modes=True (default) to run BOTH a "raw" pass and a
"postprocessed" pass that mirrors the ESMFold-prep guardrails (k-mer loop
breaking + length truncation), so you can directly compare recovery
before vs after the safety caps that may later be applied at inference
time. This is applied ONLY to the generated mask sequences, never to
ground truth.

Outputs (per mode, under <out_dir>/<raw|postprocessed>/):
    combined_metrics.csv   - one row per (checkpoint, mask)
    summary_metrics.csv    - one row per checkpoint, aggregate stats
    per_mask_tag_metrics.csv - one row per (checkpoint, mask tag)
    plots/<checkpoint>_len_diff_hist.png
    plots/<checkpoint>_identity_hist.png
    plots/<checkpoint>_substitution_breakdown.png
    plots/<checkpoint>_gt_vs_gen_len_scatter.png
    plots/all_checkpoints_identity_hist.png   (overlaid comparison)
    plots/all_checkpoints_len_diff_hist.png   (overlaid comparison)
    alignment_examples/<checkpoint>_<mode>_alignment_examples.txt
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import Align
from Bio.Align import substitution_matrices

MASK_PATTERN = re.compile(r'<M([1-9]|[1-2][0-9]|30)>')

# Semantic labels for mask tags under specific eval tasks. Keys must match
# the full tag string as extracted by parse_masked_sequence (e.g. "<M1>").
# Tags not present in the mapping fall back to their raw tag name.
TASK_MASK_LABELS = {
    "antigen": {"<M1>": "HCDR1", "<M2>": "HCDR2", "<M3>": "HCDR3"},
}

# Mirrors the kmer-loop-breaking heuristic from the ESMFold-prep pipeline:
# collapses a run of >=20 repeats of a 1-20mer down to 5 repeats.
KMER_LOOP_PATTERN = re.compile(r'([A-Z]{1,20})\1{19,}')

# Loaded once at module level so compute_alignment_stats can classify
# each aligned column as conservative (positive BLOSUM score) or
# non-conservative (zero or negative) without re-loading per call.
BLOSUM62 = substitution_matrices.load("BLOSUM62")


def make_aligner():
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"
    return aligner


ALIGNER = make_aligner()


def detect_kmer_loop(sequence):
    """Returns True if the sequence contains a degenerate repeated-kmer run
    (a common greedy-decoding failure mode where the model doesn't know
    when to stop). Detection-only, does not modify the sequence."""
    return KMER_LOOP_PATTERN.search(sequence) is not None


def break_kmer_loops(sequence):
    """Collapse degenerate repeated-kmer runs down to 5 repeats, mirroring
    the inference-time safety guardrail."""
    match = KMER_LOOP_PATTERN.search(sequence)
    if match:
        start_idx = match.start()
        kmer = match.group(1)
        sequence = sequence[:start_idx] + (kmer * 5)
    return sequence


def postprocess_mask_seq(seq, max_infill_length):
    """Apply the same guardrails the ESMFold-prep pipeline would apply
    to a generated infill span: kmer-loop breaking, then truncation."""
    seq = break_kmer_loops(seq)
    if max_infill_length is not None and len(seq) > max_infill_length:
        seq = seq[:max_infill_length]
    return seq


def parse_masked_sequence(raw_seq):
    """
    Parses 'SCAFFOLD<M1>...<SEP><M1>answer1<M2>answer2...'
    Returns (scaffold, {tag: answer_seq}).
    Returns (None, {}) if no <SEP> is present (not an infilling sequence).
    """
    if raw_seq is None or '<SEP>' not in raw_seq:
        return None, {}

    splits = raw_seq.split('<SEP>')
    scaffold = splits[0]
    answers_str = '<SEP>'.join(splits[1:])  # in case of stray extra <SEP>s

    tags_found = list(MASK_PATTERN.finditer(answers_str))
    extracted = {}
    for i, match in enumerate(tags_found):
        tag = match.group()
        start = match.end()
        end = tags_found[i + 1].start() if i + 1 < len(tags_found) else len(answers_str)
        seq = re.sub(r'[^A-Z]', '', answers_str[start:end])
        if tag not in extracted:
            extracted[tag] = seq  # keep empty strings too -- treated as failed mask
    return scaffold, extracted


def load_seqs(path):
    """Load a design.txt / target.txt file into {id: raw_seq}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            out[int(parts[0])] = parts[1]
    return out


def compute_alignment_stats(gt_seq, gen_seq):
    """
    Global BLOSUM62 alignment of gen_seq vs gt_seq.
    Returns a dict with identity, similarity, and per-column breakdown rates,
    all normalized by len(gt_seq) (recall-style).

    Column classification after alignment:
      identical        - same amino acid (x == y, x != '-')
      conservative     - different AAs, but BLOSUM62 score > 0
                         (e.g. I/V, D/E -- biochemically similar)
      nonconservative  - different AAs, BLOSUM62 score <= 0
                         (e.g. I/D -- biochemically dissimilar)
      gap              - one side is '-' (insertion or deletion)

    Derived summary metrics:
      identity    = identical / len(gt_seq)
      similarity  = (identical + conservative) / len(gt_seq)
                    "soft recovery" that gives partial credit for
                    conservative substitutions
    """
    if len(gt_seq) == 0:
        return None  # malformed/empty ground-truth mask, exclude
    if len(gen_seq) == 0:
        return {
            "identity": 0.0, "similarity": 0.0,
            "conservative_rate": 0.0, "nonconservative_rate": 0.0,
            "gap_rate": 0.0,
        }

    aln = ALIGNER.align(gt_seq, gen_seq)[0]
    a, b = aln[0], aln[1]

    identical = 0
    conservative = 0
    nonconservative = 0
    gaps = 0

    for x, y in zip(a, b):
        if x == '-' or y == '-':
            gaps += 1
        elif x == y:
            identical += 1
        else:
            try:
                score = BLOSUM62[x][y]
            except KeyError:
                score = -1  # treat unknown AA pairs as non-conservative
            if score > 0:
                conservative += 1
            else:
                nonconservative += 1

    n = len(gt_seq)
    return {
        "identity":             identical / n,
        "similarity":           (identical + conservative) / n,
        "conservative_rate":    conservative / n,
        "nonconservative_rate": nonconservative / n,
        "gap_rate":             gaps / n,
    }


def evaluate_checkpoint(ckpt_dir, apply_postprocessing, max_infill_length):
    design_path = os.path.join(ckpt_dir, "design.txt")
    target_path = os.path.join(ckpt_dir, "target.txt")
    designs = load_seqs(design_path)
    targets = load_seqs(target_path)

    mask_records = []
    seq_records = []
    gt_masks_by_id = {}   # orig_id -> {tag: gt_seq}  for alignment examples
    gen_masks_by_id = {}  # orig_id -> {tag: raw gen_seq} (pre-postprocessing)

    for orig_id, target_raw in targets.items():
        _, gt_masks = parse_masked_sequence(target_raw)
        if not gt_masks:
            continue  # autoregressive example (no <SEP>/<M#> in target), skip

        design_raw = designs.get(orig_id)
        _, gen_masks = parse_masked_sequence(design_raw) if design_raw is not None else (None, {})

        # Detect kmer-loop degeneracy on the RAW generated sequence, before
        # any postprocessing is applied. This lets us report what fraction
        # of generations were degenerate regardless of which mode we're in.
        kmer_triggered_by_tag = {tag: detect_kmer_loop(seq) for tag, seq in gen_masks.items()}

        # Store raw (pre-postprocessing) masks for alignment examples.
        # save_alignment_examples re-applies postprocessing itself so it can
        # show the right sequence for whichever mode is being reported.
        gt_masks_by_id[orig_id] = gt_masks
        gen_masks_by_id[orig_id] = dict(gen_masks)  # snapshot before postprocessing

        if apply_postprocessing:
            gen_masks = {tag: postprocess_mask_seq(seq, max_infill_length)
                         for tag, seq in gen_masks.items()}

        total_matches = 0.0
        total_gt_len = 0
        per_mask_identities = []
        per_mask_similarities = []

        for tag, gt_seq in gt_masks.items():
            gen_seq = gen_masks.get(tag, "")
            stats = compute_alignment_stats(gt_seq, gen_seq)
            if stats is None:
                continue

            total_matches += stats["identity"] * len(gt_seq)
            total_gt_len += len(gt_seq)
            per_mask_identities.append(stats["identity"])
            per_mask_similarities.append(stats["similarity"])

            mask_records.append({
                "orig_id": orig_id,
                "mask_tag": tag,
                "gt_len": len(gt_seq),
                "gen_len": len(gen_seq),
                "len_diff": len(gen_seq) - len(gt_seq),
                "identity": stats["identity"],
                "similarity": stats["similarity"],
                "conservative_rate": stats["conservative_rate"],
                "nonconservative_rate": stats["nonconservative_rate"],
                "gap_rate": stats["gap_rate"],
                "mask_missing_in_design": tag not in gen_masks,
                "kmer_loop_triggered": kmer_triggered_by_tag.get(tag, False),
            })

        if total_gt_len > 0:
            seq_records.append({
                "orig_id": orig_id,
                "num_masks": len(gt_masks),
                "weighted_recovery": total_matches / total_gt_len,
                "unweighted_recovery": float(np.mean(per_mask_identities)) if per_mask_identities else None,
                "unweighted_similarity": float(np.mean(per_mask_similarities)) if per_mask_similarities else None,
            })

    return pd.DataFrame(mask_records), pd.DataFrame(seq_records), gt_masks_by_id, gen_masks_by_id


def summarize(mask_df, seq_df, label):
    return {
        "checkpoint": label,
        "n_sequences": len(seq_df),
        "n_masks": len(mask_df),
        # --- identity (strict: exact AA match only) ---
        "mean_weighted_recovery":    seq_df["weighted_recovery"].mean() if len(seq_df) else None,
        "mean_unweighted_recovery":  seq_df["unweighted_recovery"].mean() if len(seq_df) else None,
        "mean_per_mask_identity":    mask_df["identity"].mean() if len(mask_df) else None,
        "median_per_mask_identity":  mask_df["identity"].median() if len(mask_df) else None,
        # --- similarity (soft: identity + conservative substitutions) ---
        "mean_unweighted_similarity": seq_df["unweighted_similarity"].mean() if len(seq_df) else None,
        "mean_per_mask_similarity":   mask_df["similarity"].mean() if len(mask_df) else None,
        "median_per_mask_similarity": mask_df["similarity"].median() if len(mask_df) else None,
        # --- substitution breakdown (rates over aligned gt length) ---
        "mean_conservative_rate":     mask_df["conservative_rate"].mean() if len(mask_df) else None,
        "mean_nonconservative_rate":  mask_df["nonconservative_rate"].mean() if len(mask_df) else None,
        "mean_gap_rate":              mask_df["gap_rate"].mean() if len(mask_df) else None,
        # identity - similarity gap: how much of recovery is conservative subs
        "mean_identity_similarity_gap": (
            (mask_df["similarity"] - mask_df["identity"]).mean() if len(mask_df) else None
        ),
        # --- length diagnostics ---
        "mean_len_diff":   mask_df["len_diff"].mean() if len(mask_df) else None,
        "median_len_diff": mask_df["len_diff"].median() if len(mask_df) else None,
        "std_len_diff":    mask_df["len_diff"].std() if len(mask_df) else None,
        "mean_gt_len":     mask_df["gt_len"].mean() if len(mask_df) else None,
        "mean_gen_len":    mask_df["gen_len"].mean() if len(mask_df) else None,
        # --- failure mode flags ---
        "pct_masks_missing_in_design":   mask_df["mask_missing_in_design"].mean() if len(mask_df) else None,
        "pct_masks_kmer_loop_triggered": mask_df["kmer_loop_triggered"].mean() if len(mask_df) else None,
    }


def summarize_by_mask_tag(mask_df, task_labels=None):
    """
    One row per (checkpoint, mask_tag), e.g. so you can see recovery for
    M1/M2/M3 individually instead of only pooled across all masks.

    Built directly from mask-level rows (not seq_df), since weighted/
    unweighted per-sequence recovery mixes tags together and isn't
    meaningful once you're slicing by a single tag.

    task_labels: optional dict mapping tag -> human-readable region name
    (e.g. TASK_MASK_LABELS["antigen"]), added as a "region" column.
    Tags not in the mapping just keep their raw tag as the region.
    """
    if len(mask_df) == 0:
        return pd.DataFrame()

    rows = []
    for (ckpt, tag), sub in mask_df.groupby(["checkpoint", "mask_tag"]):
        region = task_labels.get(tag, tag) if task_labels else tag
        rows.append({
            "checkpoint": ckpt,
            "mask_tag": tag,
            "region": region,
            "n_masks": len(sub),
            "mean_identity":              sub["identity"].mean(),
            "median_identity":            sub["identity"].median(),
            "mean_similarity":            sub["similarity"].mean(),
            "median_similarity":          sub["similarity"].median(),
            "mean_conservative_rate":     sub["conservative_rate"].mean(),
            "mean_nonconservative_rate":  sub["nonconservative_rate"].mean(),
            "mean_gap_rate":              sub["gap_rate"].mean(),
            "mean_len_diff":              sub["len_diff"].mean(),
            "median_len_diff":            sub["len_diff"].median(),
            "std_len_diff":               sub["len_diff"].std(),
            "mean_gt_len":                sub["gt_len"].mean(),
            "mean_gen_len":               sub["gen_len"].mean(),
            "pct_masks_missing_in_design":   sub["mask_missing_in_design"].mean(),
            "pct_masks_kmer_loop_triggered": sub["kmer_loop_triggered"].mean(),
        })

    return pd.DataFrame(rows).sort_values(["checkpoint", "mask_tag"]).reset_index(drop=True)


def make_plots(mask_df, plots_dir, mode_label):
    os.makedirs(plots_dir, exist_ok=True)

    for ckpt, sub in mask_df.groupby("checkpoint"):
        # Length difference histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sub["len_diff"], bins=30, color="#4C72B0", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("gen_len - gt_len")
        ax.set_ylabel("count")
        ax.set_title(f"{ckpt} [{mode_label}]\nMask length difference (gen - gt)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{ckpt}_len_diff_hist.png"), dpi=150)
        plt.close(fig)

        # Identity and similarity histograms, stacked for comparison
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        ax_top.hist(sub["identity"], bins=30, range=(0, 1), color="#55A868", edgecolor="black")
        ax_top.set_ylabel("count")
        ax_top.set_title(f"{ckpt} [{mode_label}]\nIdentity (strict - exact AA match)")
        ax_bot.hist(sub["similarity"], bins=30, range=(0, 1), color="#4C72B0", edgecolor="black")
        ax_bot.set_xlabel("per-mask score")
        ax_bot.set_ylabel("count")
        ax_bot.set_title("Similarity (identity + conservative substitutions)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{ckpt}_identity_hist.png"), dpi=150)
        plt.close(fig)

        # Stacked substitution breakdown: what fraction of aligned gt residues
        # are identical, conservative, non-conservative, or gapped on average.
        categories = ["identical", "conservative", "nonconservative", "gap"]
        col_map = {
            "identical":       "identity",
            "conservative":    "conservative_rate",
            "nonconservative": "nonconservative_rate",
            "gap":             "gap_rate",
        }
        colors = ["#55A868", "#4C72B0", "#C44E52", "#8C8C8C"]
        means = [sub[col_map[c]].mean() for c in categories]

        fig, ax = plt.subplots(figsize=(5, 4))
        left = 0.0
        for cat, mean, color in zip(categories, means, colors):
            ax.barh(0, mean, left=left, color=color, label=f"{cat} ({mean:.2%})", height=0.4)
            left += mean
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of aligned gt residues")
        ax.set_yticks([])
        ax.set_title(f"{ckpt} [{mode_label}]\nMean substitution breakdown")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{ckpt}_substitution_breakdown.png"), dpi=150)
        plt.close(fig)

        # gt_len vs gen_len scatter
        fig, ax = plt.subplots(figsize=(5, 5))
        max_len = max(sub["gt_len"].max(), sub["gen_len"].max())
        ax.plot([0, max_len], [0, max_len], color="red", linestyle="--", linewidth=1, label="y = x")
        ax.scatter(sub["gt_len"], sub["gen_len"], alpha=0.4, s=15, color="#4C72B0")
        ax.set_xlabel("ground-truth mask length")
        ax.set_ylabel("generated mask length")
        ax.set_title(f"{ckpt} [{mode_label}]\nGT vs generated mask length")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{ckpt}_gt_vs_gen_len_scatter.png"), dpi=150)
        plt.close(fig)

    # Overlaid cross-checkpoint comparison plots
    checkpoints = sorted(mask_df["checkpoint"].unique())

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for ckpt in checkpoints:
        sub = mask_df[mask_df["checkpoint"] == ckpt]
        ax_top.hist(sub["identity"], bins=30, range=(0, 1), alpha=0.5, label=ckpt)
        ax_bot.hist(sub["similarity"], bins=30, range=(0, 1), alpha=0.5, label=ckpt)
    ax_top.set_ylabel("count")
    ax_top.set_title(f"Identity (strict) across checkpoints [{mode_label}]")
    ax_top.legend(fontsize=8)
    ax_bot.set_xlabel("per-mask score")
    ax_bot.set_ylabel("count")
    ax_bot.set_title(f"Similarity (identity + conservative) across checkpoints [{mode_label}]")
    ax_bot.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "all_checkpoints_identity_hist.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for ckpt in checkpoints:
        sub = mask_df[mask_df["checkpoint"] == ckpt]
        ax.hist(sub["len_diff"], bins=30, alpha=0.5, label=ckpt)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("gen_len - gt_len")
    ax.set_ylabel("count")
    ax.set_title(f"Length difference across checkpoints [{mode_label}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "all_checkpoints_len_diff_hist.png"), dpi=150)
    plt.close(fig)


def format_blast_alignment(gt_seq, gen_seq, line_width=60):
    """
    Formats a pairwise alignment in BLAST style for human reading:

        Query  1   MAILVKDEST  10
                   ||||:  .|.
        Sbjct  1   MAILAKGEST  10

    Symbols:
        |   identical
        :   conservative substitution  (BLOSUM62 > 0)
        .   non-conservative mismatch  (BLOSUM62 <= 0)
            gap (space)

    Returns a formatted string. gt_seq is Query, gen_seq is Sbjct.
    """
    if len(gen_seq) == 0:
        return "  (no sequence generated for this mask)\n"

    aln = ALIGNER.align(gt_seq, gen_seq)[0]
    a, b = aln[0], aln[1]  # aligned gt, aligned gen (may contain '-')

    symbols = []
    for x, y in zip(a, b):
        if x == '-' or y == '-':
            symbols.append(' ')
        elif x == y:
            symbols.append('|')
        else:
            try:
                score = BLOSUM62[x][y]
            except KeyError:
                score = -1
            symbols.append(':' if score > 0 else '.')
    symbols = ''.join(symbols)

    # Chunk into line_width blocks with position counters, skipping gaps
    # in position count (same convention as BLAST).
    lines = []
    q_pos = 1   # position in gt_seq (gaps don't advance)
    s_pos = 1   # position in gen_seq

    for chunk_start in range(0, len(a), line_width):
        chunk_end = min(chunk_start + line_width, len(a))

        q_chunk = a[chunk_start:chunk_end]
        s_chunk = b[chunk_start:chunk_end]
        m_chunk = symbols[chunk_start:chunk_end]

        q_end = q_pos + len(q_chunk.replace('-', '')) - 1
        s_end = s_pos + len(s_chunk.replace('-', '')) - 1

        lines.append(f"  Query  {q_pos:>4}  {q_chunk}  {q_end}")
        lines.append(f"               {m_chunk}")
        lines.append(f"  Sbjct  {s_pos:>4}  {s_chunk}  {s_end}")
        lines.append("")

        q_pos = q_end + 1
        s_pos = s_end + 1

    return '\n'.join(lines)


def save_alignment_examples(combined_mask_df, designs_by_ckpt, targets_by_ckpt,
                            out_dir, mode_label, apply_postprocessing, max_infill_length):
    """
    For each checkpoint, picks the single best and single worst mask by
    identity score and writes a BLAST-style alignment report to a .txt file
    under <out_dir>/alignment_examples/.
    """
    examples_dir = os.path.join(out_dir, "alignment_examples")
    os.makedirs(examples_dir, exist_ok=True)

    for ckpt, sub in combined_mask_df.groupby("checkpoint"):
        if sub.empty:
            continue

        best_row = sub.loc[sub["identity"].idxmax()]
        worst_row = sub.loc[sub["identity"].idxmin()]

        lines = []
        lines.append("=" * 70)
        lines.append(f"Checkpoint : {ckpt}")
        lines.append(f"Mode       : {mode_label}")
        lines.append("=" * 70)

        for label, row in [("BEST", best_row), ("WORST", worst_row)]:
            orig_id = int(row["orig_id"])
            mask_tag = row["mask_tag"]

            gt_masks = targets_by_ckpt[ckpt].get(orig_id, {})
            gen_masks = designs_by_ckpt[ckpt].get(orig_id, {})

            gt_seq = gt_masks.get(mask_tag, "")
            gen_seq = gen_masks.get(mask_tag, "")

            if apply_postprocessing:
                gen_seq = postprocess_mask_seq(gen_seq, max_infill_length)

            lines.append("")
            lines.append(f"  [{label} EXAMPLE]")
            lines.append(f"  Sequence ID : {orig_id}")
            lines.append(f"  Mask        : {mask_tag}")
            lines.append(f"  Identity    : {row['identity']:.3f}")
            lines.append(f"  Similarity  : {row['similarity']:.3f}")
            lines.append(f"  GT length   : {row['gt_len']}  |  Gen length: {row['gen_len']}  |  Diff: {row['len_diff']:+d}")
            if row["kmer_loop_triggered"]:
                lines.append("  WARNING: kmer loop detected in raw generation")
            lines.append("")
            lines.append(format_blast_alignment(gt_seq, gen_seq))

        lines.append("=" * 70)
        lines.append("")

        out_path = os.path.join(examples_dir, f"{ckpt}_{mode_label}_alignment_examples.txt")
        with open(out_path, "w") as f:
            f.write('\n'.join(lines))

    print(f"  Alignment examples written to {examples_dir}/")


def run(checkpoint_dirs, out_dir, apply_postprocessing, max_infill_length, task=None):
    mode_label = "postprocessed" if apply_postprocessing else "raw"
    run_out_dir = os.path.join(out_dir, mode_label)
    plots_dir = os.path.join(run_out_dir, "plots")
    os.makedirs(run_out_dir, exist_ok=True)

    all_mask_dfs = []
    all_summaries = []
    all_gt_masks = {}   # ckpt_label -> gt_masks_by_id
    all_gen_masks = {}  # ckpt_label -> gen_masks_by_id

    for ckpt_dir in checkpoint_dirs:
        label = os.path.basename(ckpt_dir.rstrip('/'))
        print(f"[{mode_label}] Processing {label} ...")

        if not os.path.isdir(ckpt_dir):
            print(f"  Directory not found, skipping: {ckpt_dir}")
            continue

        mask_df, seq_df, gt_masks_by_id, gen_masks_by_id = evaluate_checkpoint(
            ckpt_dir, apply_postprocessing, max_infill_length)
        if len(mask_df) == 0:
            print(f"  No infilling examples found in {ckpt_dir}")
            continue

        mask_df["checkpoint"] = label
        all_mask_dfs.append(mask_df)
        all_summaries.append(summarize(mask_df, seq_df, label))
        all_gt_masks[label] = gt_masks_by_id
        all_gen_masks[label] = gen_masks_by_id
        print(f"  -> {len(seq_df)} infilling sequences, {len(mask_df)} masks evaluated")

    if not all_mask_dfs:
        print(f"[{mode_label}] No data processed across any checkpoint directory.")
        return None, None

    combined_mask_df = pd.concat(all_mask_dfs, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    combined_mask_df.to_csv(os.path.join(run_out_dir, "combined_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(run_out_dir, "summary_metrics.csv"), index=False)

    task_labels = TASK_MASK_LABELS.get(task) if task else None
    per_tag_df = summarize_by_mask_tag(combined_mask_df, task_labels)
    per_tag_df.to_csv(os.path.join(run_out_dir, "per_mask_tag_metrics.csv"), index=False)
    print(f"[{mode_label}] === Per-mask-tag summary ===")
    print(per_tag_df.to_string(index=False))
    print()

    if task_labels:
        hcdr_df = per_tag_df[per_tag_df["mask_tag"].isin(task_labels.keys())].copy()
        hcdr_df.to_csv(os.path.join(run_out_dir, "hcdr_metrics.csv"), index=False)

    make_plots(combined_mask_df, plots_dir, mode_label)
    save_alignment_examples(combined_mask_df, all_gen_masks, all_gt_masks,
                            run_out_dir, mode_label, apply_postprocessing, max_infill_length)

    print(f"\n[{mode_label}] Wrote per-mask data, summary, and plots to {run_out_dir}\n")
    print(f"=== Summary [{mode_label}] ===")
    print(summary_df.to_string(index=False))
    print()

    return combined_mask_df, summary_df


def main(checkpoint_dirs, out_dir=".", max_infill_length=50, run_both_modes=True, task=None):
    """
    run_both_modes=True runs both 'raw' and 'postprocessed' passes so you can
    directly compare recovery before/after the inference-time safety caps
    would be applied. Set to False to only run the raw pass.

    task=None runs generically. task="antigen" additionally labels
    <M1>/<M2>/<M3> as HCDR1/HCDR2/HCDR3 in per_mask_tag_metrics.csv and
    writes a filtered hcdr_metrics.csv with just those three masks.
    """
    run(checkpoint_dirs, out_dir, apply_postprocessing=False,
        max_infill_length=max_infill_length, task=task)
    if run_both_modes:
        run(checkpoint_dirs, out_dir, apply_postprocessing=True,
            max_infill_length=max_infill_length, task=task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Infilling Recovery Metrics")
    parser.add_argument("--ckpt_dirs", nargs='+', required=True,
                        help="Directories containing design.txt and target.txt")
    parser.add_argument("--out_dir", default="./infill_metrics_results",
                        help="Directory to save metrics and plots")
    parser.add_argument("--max_infill_length", type=int, default=50,
                        help="Max length for infill sequences")
    parser.add_argument("--task", choices=["antigen"], default=None,
                        help="Optional task label. 'antigen' maps <M1>/<M2>/<M3> to "
                             "HCDR1/HCDR2/HCDR3 in per_mask_tag_metrics.csv and writes "
                             "a filtered hcdr_metrics.csv.")
    args = parser.parse_args()

    main(
        checkpoint_dirs=args.ckpt_dirs,
        out_dir=args.out_dir,
        max_infill_length=args.max_infill_length,
        run_both_modes=True,
        task=args.task,
    )
