from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from Bio import Phylo
from numpy.typing import NDArray

from sweep.models import PCCIResult, PMPGResult


def costquebra(
    z: Sequence[int | float] | NDArray[np.int32 | np.int64 | np.float64],
    mode: str = "new",
) -> float:
    """Calculates taxon breakage cost matching R's costquebra function."""
    del mode
    z_arr = np.asarray(z, dtype=np.int32)
    ones_idx = np.where(z_arr == 1)[0]
    n = len(ones_idx)
    if n <= 1:
        return 1.0

    xo = ones_idx[0]
    xe = ones_idx[-1]
    slice_ones = np.where(z_arr[xo : xe + 1] == 1)[0]
    adjacent_pairs = int(np.sum(np.diff(slice_ones) == 1))
    return float((1.0 + adjacent_pairs) / float(n))


def _load_metadata(
    mt: str | Path | pl.DataFrame | dict[str, str] | None,
) -> dict[str, str] | None:
    if mt is None:
        return None
    if isinstance(mt, dict):
        return mt
    if isinstance(mt, (str, Path)):
        df = pl.read_csv(mt)
    elif isinstance(mt, pl.DataFrame):
        df = mt
    else:
        msg = f"Unsupported metadata type: {type(mt)}"
        raise TypeError(msg)

    cols = df.columns
    if "fileName" in cols and "family" in cols:
        return dict(
            zip(
                df["fileName"].cast(pl.String).to_list(),
                df["family"].cast(pl.String).to_list(),
                strict=True,
            )
        )
    return dict(
        zip(
            df[cols[0]].cast(pl.String).to_list(),
            df[cols[1]].cast(pl.String).to_list(),
            strict=True,
        )
    )


def _load_tree(tr: str | Path | Any) -> Any:
    if isinstance(tr, (str, Path)):
        return Phylo.read(str(tr), "newick")
    return tr


def pcci(
    tr: str | Path | Any,
    mt: str | Path | pl.DataFrame | dict[str, str] | None = None,
) -> PCCIResult:
    """PhyloTaxonomic Consistency Cophenetic Index (PCCI) evaluation."""
    tree = copy.deepcopy(_load_tree(tr))
    metadata = _load_metadata(mt)

    terminals = tree.get_terminals()
    if metadata is not None:
        for term in terminals:
            if term.name in metadata:
                term.name = metadata[term.name]

    for clade in tree.find_clades():
        clade.branch_length = 1.0

    taxa = [str(t.name) if t.name is not None else "" for t in terminals]
    unique_taxa: list[str] = []
    for tx in taxa:
        if tx and tx not in unique_taxa:
            unique_taxa.append(tx)

    n_term = len(terminals)
    dmat = np.zeros((n_term, n_term), dtype=np.float64)
    for i in range(n_term):
        for j in range(n_term):
            dmat[i, j] = float(tree.distance(terminals[i], terminals[j]))

    costs: list[float] = []
    for label in unique_taxa:
        idx = [i for i, tx in enumerate(taxa) if tx == label]
        submat = dmat[idx, :]
        binvec = np.zeros(len(taxa), dtype=np.float64)
        binvec[idx] = 1.0

        dist_flat = submat.flatten()
        binvec_rep = np.tile(binvec, len(idx))

        order = np.argsort(dist_flat, kind="stable")
        dist_sorted = dist_flat[order]
        bin_sorted = binvec_rep[order]

        last_one = int(np.max(np.where(bin_sorted == 1.0)[0]))
        dist_trunc = dist_sorted[: last_one + 1]
        bin_trunc = bin_sorted[: last_one + 1]

        for u in np.unique(dist_trunc):
            m_idx = np.where(dist_trunc == u)[0]
            bin_trunc[m_idx] = np.sort(bin_trunc[m_idx])[::-1]

        cost = costquebra(bin_trunc)
        costs.append(cost)

    tab = pl.DataFrame(
        {
            "taxa": unique_taxa,
            "cost": costs,
        }
    )
    mean_cost = float(np.mean(costs)) if costs else 0.0
    return PCCIResult(tab=tab, mean=mean_cost)


def _is_monophyletic(current_tree: Any, target_taxon: str) -> bool:
    matching_tips = [t for t in current_tree.get_terminals() if t.name == target_taxon]
    all_tips = current_tree.get_terminals()
    if len(matching_tips) <= 1 or len(matching_tips) == len(all_tips):
        return True

    mrca = current_tree.common_ancestor(matching_tips)
    descendant_names = [t.name for t in mrca.get_terminals()]
    return all(name == target_taxon for name in descendant_names)


def pmpg(
    tr: str | Path | Any,
    mt: str | Path | pl.DataFrame | dict[str, str] | None = None,
) -> PMPGResult:
    """Percentage of Mono or Paraphyletic Groups (PMPG) evaluation."""
    tree = copy.deepcopy(_load_tree(tr))
    metadata = _load_metadata(mt)

    terminals = tree.get_terminals()
    if metadata is not None:
        for term in terminals:
            if term.name in metadata:
                term.name = metadata[term.name]

    taxa = [str(t.name) if t.name is not None else "" for t in terminals]
    unique_taxa: list[str] = []
    for tx in taxa:
        if tx and tx not in unique_taxa:
            unique_taxa.append(tx)

    n_taxa = len(unique_taxa)
    if n_taxa == 0:
        empty_tab = pl.DataFrame(
            {"taxa": [], "mono": [], "para": []},
            schema={
                "taxa": pl.String,
                "mono": pl.Boolean,
                "para": pl.Boolean,
            },
        )
        return PMPGResult(
            tab=empty_tab,
            perc_mono=0.0,
            perc_para=0.0,
            metric=0.0,
        )

    current_tree = copy.deepcopy(tree)
    subtaxa = list(unique_taxa)
    past_mono = [_is_monophyletic(current_tree, tx) for tx in subtaxa]
    rounds: list[list[bool]] = [past_mono]

    have_non_mono = any(not m for m in past_mono)
    while have_non_mono:
        mono_in_current_round = [subtaxa[i] for i, m in enumerate(past_mono) if m]
        if not mono_in_current_round:
            break

        tips_to_drop = [t for t in current_tree.get_terminals() if t.name in mono_in_current_round]
        for tip in tips_to_drop:
            current_tree.prune(tip)

        subtaxa = [subtaxa[i] for i, m in enumerate(past_mono) if not m]
        if not subtaxa or not current_tree.get_terminals():
            break

        current_mono = [_is_monophyletic(current_tree, tx) for tx in subtaxa]
        rounds.append(current_mono)
        if not any(current_mono):
            break
        past_mono = current_mono

    mono_round1 = rounds[0]
    para_flags: dict[str, bool] = {tx: False for tx in unique_taxa}

    if len(rounds) > 1:
        # Subsequent rounds indicate paraphyly
        sub_names = list(unique_taxa)
        for r in range(1, len(rounds)):
            round_flags = rounds[r]
            # Subtaxa present in this round were those False in previous round
            sub_names = [sub_names[i] for i, flag in enumerate(rounds[r - 1]) if not flag]
            for tx, flag in zip(sub_names, round_flags, strict=True):
                if flag:
                    para_flags[tx] = True

    tab = pl.DataFrame(
        {
            "taxa": unique_taxa,
            "mono": mono_round1,
            "para": [para_flags[tx] for tx in unique_taxa],
        }
    )

    n_mono = sum(1 for m in mono_round1 if m)
    perc_mono = 100.0 * float(n_mono) / float(n_taxa) if n_taxa > 0 else 0.0

    n_para = sum(1 for tx in unique_taxa if para_flags[tx])
    perc_para = 100.0 * float(n_para) / float(n_taxa) if n_taxa > 0 else 0.0

    metric = perc_mono + perc_para

    return PMPGResult(
        tab=tab,
        perc_mono=perc_mono,
        perc_para=perc_para,
        metric=metric,
    )
