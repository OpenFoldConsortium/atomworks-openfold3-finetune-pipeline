"""Template utilities: ColabFold template hits -> the layouts OF3 reads.

Flow: generate-msas writes {msa_dir}/templates/<shard>/<rep_id>.m8; fetch-templates
(fetch_templates + reorganize_template_alignments_for_of3) writes
{msa_dir}/template_structures/<pdb_id>.cif and
{msa_dir}/of3_template_alignments/<rep_id>/colabfold_template.m8 (hits remapped to
label chain IDs, which OF3's template preprocessing expects). preprocess-templates
(preprocess_templates_for_of3) drops hits failing OF3's template sequence filters, then
runs OF3's train-mode template preprocessing into {msa_dir}/of3_template_data/ and
writes template_ids into dataset_cache.json.
find_template_ids reads that cache for finetune-prep --template-cache-dir.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import biotite.sequence as bseq
import biotite.sequence.align as balign
import biotite.structure.io.pdbx as pdbx
import numpy as np
from atomworks.ml.preprocessing.msa.finding import find_template_alignments
from atomworks.ml.preprocessing.msa.template_structures import (
    fetch_template_structures_from_m8_file,
)
from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.io.sequence.template import TemplateHit
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from openfold3.core.data.primitives.sequence.template import check_sequence
from requests.exceptions import RequestException
from openfold3.core.data.primitives.structure.metadata import (
    get_author_to_label_chain_ids,
)
from pydantic import BaseModel, ConfigDict

from aw_of3_finetune.constants import TEMPLATE_CUTOFF_FILENAME
from aw_of3_finetune.msa import (
    alignment_rep_id,
    chain_canonical_sequences,
    collect_msa_queries,
)

logger = logging.getLogger(__name__)

DOWNLOAD_ATTEMPTS = 4
FIRST_RETRY_WAIT_SECONDS = 5  # then 10 and 20

# Layout under --msa-dir, next to generate-msas' unpaired/ paired/ templates/.
TEMPLATE_ALIGNMENTS_DIRNAME = "of3_template_alignments"  # <rep_id>/colabfold_template.m8
TEMPLATE_ALIGNMENT_FILENAME = "colabfold_template.m8"
TEMPLATE_STRUCTURES_DIRNAME = "template_structures"  # flat <pdb_id>.cif
# OF3's template preprocessing output: template_cache/<rep_id>.npz and
# template_structure_arrays/<pdb_id>/<pdb_id>_<chain>.npz.
TEMPLATE_DATA_DIRNAME = "of3_template_data"
# Under TEMPLATE_DATA_DIRNAME: the .m8 files OF3 reads, after the sequence filters.
FILTERED_TEMPLATE_ALIGNMENTS_DIRNAME = "filtered_template_alignments"
SEQUENCE_FILTER_SETTINGS_FILENAME = "template_sequence_filter.json"
SEQUENCE_FILTER_REPORT_FILENAME = "template_sequence_filter.tsv"

# Local alignment used to find each hit's aligned template segment (BLOSUM62, gap 10/1).
_SUBSTITUTION_MATRIX = balign.SubstitutionMatrix.std_protein_matrix()
_GAP_PENALTY = (-10, -1)
_PROTEIN_ALPHABET = set(_SUBSTITUTION_MATRIX.get_alphabet1())


class TemplateSequenceFilterSettings(BaseModel):
    """OF3's template sequence filters (AF3 SI 2.4); defaults are OF3's check_sequence ones.

    Attributes:
        enabled: Apply the filters at all.
        max_subseq: Drop hits whose aligned template segment is an exact substring of
            the query and longer than this fraction of it.
        min_align: Drop hits aligned to at most this fraction of the query residues.
        min_len: Drop hits whose aligned template segment is shorter than this.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_subseq: float = 0.95
    min_align: float = 0.1
    min_len: int = 10


def write_alignment_reps_fasta(
    cif_paths: Iterable[Path],
    output: Path,
) -> dict[str, str]:
    """Write a FASTA of unique polymer sequences across CIFs, headed by rep_id.

    For running a template search outside this pipeline (OF3's
    ``alignment_representatives_fasta`` format).

    Args:
        cif_paths: mmCIF files to collect polymer sequences from.
        output: FASTA path to write; parent directories are created.

    Returns:
        {rep_id: sequence} for every unique sequence written.
    """
    rep_id_to_seq: dict[str, str] = {}
    for cif_path in cif_paths:
        for seq in chain_canonical_sequences(cif_path).values():
            rep = alignment_rep_id(seq)
            if rep not in rep_id_to_seq:
                rep_id_to_seq[rep] = seq

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for rep, seq in rep_id_to_seq.items():
            f.write(f">{rep}\n{seq}\n")

    return rep_id_to_seq


def find_template_ids(rep_id: str, template_cache_dir: Path) -> list[str]:
    """Return a chain's template IDs from a pre-built OF3 template cache.

    Reads ``{template_cache_dir}/{rep_id}.npz``, ordered by e-value rank (``index``).

    Args:
        rep_id: The chain's alignment_representative_id.
        template_cache_dir: Directory of OF3 template cache entries.

    Returns:
        Template IDs of the form ``{entry_id}_{chain_id}`` (e.g. ``"1abc_A"``), or
        ``[]`` if there is no cache entry for rep_id.
    """
    cache_file = template_cache_dir / f"{rep_id}.npz"
    if not cache_file.exists():
        return []
    with np.load(cache_file, allow_pickle=True) as npz:
        items = {k: v.item() for k, v in npz.items()}
    return [k for k, _ in sorted(items.items(), key=lambda x: x[1]["index"])]


def reorganize_template_alignments_for_of3(
    output_dir: Path, msa_dir: Path, dest_root: Path
) -> tuple[int, int]:
    """Write each dataset sequence's ``.m8`` to ``{dest_root}/{rep_id}/colabfold_template.m8``.

    Hits are rewritten to label chain IDs, read from ``{msa_dir}/template_structures/``
    (run fetch_templates first). dest_root must not be of3_alignments/.

    Args:
        output_dir: finetune-prep output directory (manifest.json + cleaned_structures/).
        msa_dir: generate-msas output directory, containing ``templates/<shard>/``.
        dest_root: Root of the per-rep_id layout to write.

    Returns:
        (written, missing): files written this call (existing ones are skipped), and
        sequences without a usable ``.m8`` (no PDB70 hits, run with --no-templates,
        or no hit could be mapped to a fetched structure).
    """
    queries = collect_msa_queries(output_dir)  # rep_id -> sequence
    if not queries:
        return 0, 0
    missing, found = find_template_alignments(list(queries.values()), [msa_dir / "templates"])

    structure_dir = msa_dir / TEMPLATE_STRUCTURES_DIRNAME
    written = 0
    n_missing = len(missing)
    for rep_id, seq in queries.items():
        src = found.get(seq)
        dest = dest_root / rep_id / TEMPLATE_ALIGNMENT_FILENAME
        if src is None or dest.exists():
            continue
        m8_text = remap_m8_to_label_chain_ids(src.read_text(), structure_dir)
        if not m8_text:
            n_missing += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(m8_text)
        written += 1
    return written, n_missing


def remap_m8_to_label_chain_ids(m8_text: str, structure_dir: Path) -> str:
    """Rewrite ColabFold hit IDs from author to label chain IDs (``1fdl_L`` -> ``1fdl_A``).

    Same rules as OF3's remap_colabfold_template_chain_ids, but mapped from local
    structures instead of the RCSB API. Rows without a fetched structure are dropped
    with one summary log line; a chain missing from a fetched structure is warned.

    Args:
        m8_text: Raw ColabFold pdb70 ``.m8`` table; column 2 is ``<pdb_id>_<author chain>``.
        structure_dir: Flat directory of ``<pdb_id>.cif`` hit structures.

    Returns:
        The table with column 2 remapped; empty if no row could be mapped.
    """
    # None marks an entry whose structure wasn't fetched (expected beyond --max-hits).
    author_to_label: dict[str, dict[str, list[str]] | None] = {}
    rows = []
    n_not_fetched = 0
    for line in m8_text.splitlines():
        if not line.strip():
            continue
        columns = line.split("\t")
        pdb_id, author_chain_id = columns[1].split("_", 1)
        pdb_id = pdb_id.lower()
        if pdb_id not in author_to_label:
            cif_path = structure_dir / f"{pdb_id}.cif"
            author_to_label[pdb_id] = (
                get_author_to_label_chain_ids(_label_to_author_chain_ids(cif_path))
                if cif_path.exists()
                else None
            )
        if author_to_label[pdb_id] is None:
            n_not_fetched += 1
            continue
        labels = author_to_label[pdb_id].get(author_chain_id)
        if not labels:
            logger.warning(
                f"Dropping template hit {columns[1]}: chain {author_chain_id} not in "
                f"{pdb_id}'s author chain IDs"
            )
            continue
        columns[1] = f"{pdb_id}_{labels[0]}"
        rows.append("\t".join(columns) + "\n")
    if n_not_fetched:
        logger.info(f"Skipped {n_not_fetched} hit(s) whose structures were not fetched")
    return "".join(rows)


def _label_to_author_chain_ids(cif_path: Path) -> dict[str, str]:
    """Read polymer label -> author chain IDs, the map OF3 otherwise gets from RCSB.

    Args:
        cif_path: mmCIF with a ``pdbx_poly_seq_scheme`` category.

    Returns:
        {label_asym_id: pdb_strand_id}.
    """
    scheme = pdbx.CIFFile.read(cif_path).block["pdbx_poly_seq_scheme"]
    return {
        str(label): str(author)
        for label, author in zip(
            scheme["asym_id"].as_array(), scheme["pdb_strand_id"].as_array()
        )
    }


def materialize_template_structures(paths: dict[str, Path], dest_dir: Path) -> dict[str, Path]:
    """Gather template structures into one flat, uncompressed ``{dest_dir}/<pdb_id>.cif``.

    ``.cif.gz`` (mirror) files are decompressed, other ``.cif`` files copied; existing
    dest files, including downloads already in dest_dir, are never rewritten.

    Args:
        paths: {pdb_id: path}, as returned by atomworks' fetch_template_structures
            (mirror ``<mirror>/<id[1:3]>/<id>.cif.gz`` and downloaded ``<id>.cif`` mixed).
        dest_dir: The flat structure directory OF3 reads; created if missing.

    Returns:
        {pdb_id: path of ``<pdb_id>.cif`` in dest_dir}.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for pdb_id, src in paths.items():
        dest = dest_dir / f"{pdb_id}.cif"
        out[pdb_id] = dest
        if dest.exists():
            continue
        if src.suffix == ".gz":
            with gzip.open(src, "rb") as f_in, dest.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copyfile(src, dest)
    return out


def fails_template_sequence_filter(
    query_seq: str, template_seq: str, settings: TemplateSequenceFilterSettings
) -> bool:
    """Apply OF3's check_sequence to a template chain, aligned locally to the query.

    The local alignment stands in for the hmmsearch alignment OF3's legacy path filters.

    Args:
        query_seq: Query (dataset chain) canonical sequence.
        template_seq: Template chain canonical (SEQRES) sequence.
        settings: Filter thresholds.

    Returns:
        True if the template fails the filters and should be dropped.
    """
    query_row, template_row = _local_alignment_a3m_rows(query_seq, template_seq)
    fails, _, _ = check_sequence(
        query=TemplateHit(index=0, name="query", aligned_cols=0, hit_sequence=query_row, e_value=None),
        hit=TemplateHit(index=1, name="hit", aligned_cols=0, hit_sequence=template_row, e_value=None),
        max_subseq=settings.max_subseq,
        min_align=settings.min_align,
        min_len=settings.min_len,
    )
    return fails


def _local_alignment_a3m_rows(query_seq: str, template_seq: str) -> tuple[str, str]:
    """Locally align template to query and write both as a3m rows, as hmmsearch does.

    Args:
        query_seq: Query sequence; its row is the full, ungapped query.
        template_seq: Template sequence.

    Returns:
        (query_row, template_row): template_row has one uppercase residue or ``-`` per
        query position, with template insertions in lowercase.
    """
    query = bseq.ProteinSequence(_to_protein_alphabet(query_seq))
    template = bseq.ProteinSequence(_to_protein_alphabet(template_seq))
    alignment = balign.align_optimal(
        query, template, _SUBSTITUTION_MATRIX, gap_penalty=_GAP_PENALTY, local=True, max_number=1
    )[0]

    columns = ["-"] * len(query_seq)
    insertions = [""] * len(query_seq)  # template residues after each query position
    last_query_pos = 0
    for query_pos, template_pos in alignment.trace:
        if template_pos < 0:
            continue
        if query_pos >= 0:
            columns[query_pos] = template_seq[template_pos].upper()
            last_query_pos = query_pos
        else:
            insertions[last_query_pos] += template_seq[template_pos].lower()
    template_row = "".join(col + ins for col, ins in zip(columns, insertions))
    return query_seq, template_row


def _to_protein_alphabet(seq: str) -> str:
    """Replace letters outside biotite's protein alphabet (e.g. U, O) with X."""
    return "".join(c if c in _PROTEIN_ALPHABET else "X" for c in seq.upper())


@lru_cache(maxsize=4096)
def _template_chain_sequences(cif_path: Path) -> dict[str, str]:
    """Read a template structure's protein label-chain sequences, once per file.

    Args:
        cif_path: Template mmCIF.

    Returns:
        {label_asym_id: canonical sequence}; empty if the file does not exist.
    """
    if not cif_path.exists():
        return {}
    return chain_canonical_sequences(cif_path, protein_only=True)


def filter_template_alignment(
    m8_text: str,
    query_seq: str,
    structure_dir: Path,
    settings: TemplateSequenceFilterSettings,
) -> tuple[str, list[str]]:
    """Drop ``.m8`` rows whose template chain fails OF3's template sequence filters.

    Rows whose structure or protein chain is missing are kept; OF3 skips them itself.

    Args:
        m8_text: ``.m8`` table with label chain IDs (``<pdb_id>_<label_asym_id>``).
        query_seq: The query sequence the hits were searched with.
        structure_dir: Flat directory of ``<pdb_id>.cif`` template structures.
        settings: Filter thresholds.

    Returns:
        (kept_text, dropped_template_ids), both in input order.
    """
    if not settings.enabled:
        return m8_text, []
    kept_rows = []
    dropped = []
    for line in m8_text.splitlines():
        if not line.strip():
            continue
        template_id = line.split("\t")[1]
        pdb_id, chain_id = template_id.split("_", 1)
        template_seq = _template_chain_sequences(structure_dir / f"{pdb_id}.cif").get(chain_id)
        if template_seq and fails_template_sequence_filter(query_seq, template_seq, settings):
            dropped.append(template_id)
        else:
            kept_rows.append(line + "\n")
    return "".join(kept_rows), dropped


class TemplateDownloadError(RuntimeError):
    """RCSB could not be reached while downloading template structures."""


def _fetch_structures_with_retries(
    m8_path: Path, structure_dir: Path, max_hits: int, pdb_mirror_path: Path | None
) -> dict[str, Path]:
    """Resolve an ``.m8``'s hits, retrying network failures with growing waits.

    Atomworks skips entries RCSB refuses, such as obsolete IDs, but lets network errors
    through. Finished downloads stay on disk, so each attempt fetches only what is left.

    Args:
        m8_path: Raw ColabFold ``.m8`` file.
        structure_dir: Directory the structures download into.
        max_hits: Unique PDB entries to resolve.
        pdb_mirror_path: Local PDB mirror to read entries from before downloading.

    Returns:
        Local path of each resolved entry.

    Raises:
        TemplateDownloadError: If every attempt fails.
    """
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return fetch_template_structures_from_m8_file(
                m8_path, structure_dir, max_hits=max_hits, pdb_mirror_path=pdb_mirror_path
            )
        except RequestException as exc:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise TemplateDownloadError(
                    f"Could not download template structures from RCSB after "
                    f"{DOWNLOAD_ATTEMPTS} attempts: {exc}\n"
                    f"Finished downloads are kept in {structure_dir}. Rerun the same command "
                    "with --skip-existing to reuse the MSAs and continue."
                ) from exc
            wait = FIRST_RETRY_WAIT_SECONDS * 2 ** (attempt - 1)
            logger.warning(f"RCSB download failed, retrying in {wait} s: {exc}")
            time.sleep(wait)


@dataclass
class FetchTemplatesSummary:
    """Counts reported by fetch_templates.

    Attributes:
        n_alignments: Dataset sequences that have a ``.m8``.
        n_structures: Unique template entries now in the structure directory.
    """

    n_alignments: int
    n_structures: int


def fetch_templates(
    output_dir: Path,
    msa_dir: Path,
    max_hits: int = 50,
    pdb_mirror_path: Path | None = None,
) -> FetchTemplatesSummary:
    """Resolve the hit structures of every dataset sequence's ``.m8`` into one flat dir.

    Writes ``{msa_dir}/template_structures/<pdb_id>.cif``; entries are shared across
    sequences and never re-downloaded.

    Args:
        output_dir: finetune-prep output directory (manifest.json + cleaned_structures/).
        msa_dir: generate-msas output directory, containing ``templates/<shard>/``.
        max_hits: Unique PDB entries to resolve per sequence, best e-value first.
            Applied before the sequence and release-date filters, so keep it >= the
            number of templates that should survive (OF3's max_templates default is 20).
        pdb_mirror_path: Local PDB mirror to read entries from before downloading.
            None downloads everything from RCSB.

    Returns:
        Counts of sequences with hits and of structures resolved.

    Raises:
        TemplateDownloadError: If RCSB stays unreachable through the retries.
    """
    queries = collect_msa_queries(output_dir)
    _, found = find_template_alignments(list(queries.values()), [msa_dir / "templates"])

    structure_dir = msa_dir / TEMPLATE_STRUCTURES_DIRNAME
    resolved: dict[str, Path] = {}
    for m8_path in found.values():
        paths = _fetch_structures_with_retries(m8_path, structure_dir, max_hits, pdb_mirror_path)
        resolved.update(materialize_template_structures(paths, structure_dir))
    return FetchTemplatesSummary(n_alignments=len(found), n_structures=len(resolved))


@dataclass(frozen=True)
class PreprocessTemplatesSummary:
    """Counts reported by preprocess_templates_for_of3.

    Attributes:
        n_chains_with_templates: Dataset-cache chains with at least one template.
        date_filtered_chains: ``<pdb_id>/<label_asym_id>`` of chains that had template
            hits but kept none after the release-date filter.
        report_path: Per-chain TSV of hits vs. templates kept.
        n_sequence_filtered: Template hits dropped by the sequence filters, summed over
            alignment representatives.
        sequence_filter_report_path: TSV of every hit the sequence filters dropped.
    """

    n_chains_with_templates: int
    date_filtered_chains: list[str]
    report_path: Path
    n_sequence_filtered: int
    sequence_filter_report_path: Path


def preprocess_templates_for_of3(
    output_dir: Path,
    msa_dir: Path,
    cache_file: Path,
    *,
    min_release_date_diff: int = 60,
    max_release_date: datetime | None = None,
    template_cutoff_date: datetime | None = None,
    max_templates: int = 20,
    n_processes: int = 1,
    ccd_path: Path | None = None,
    sequence_filter: TemplateSequenceFilterSettings | None = None,
) -> PreprocessTemplatesSummary:
    """Run OF3's train-mode template preprocessing and write template_ids into cache_file.

    Hits failing OF3's template sequence filters are dropped first. Writes the template
    cache, structure arrays and per-chain reports under ``{msa_dir}/of3_template_data/``.

    Args:
        output_dir: finetune-prep output directory (manifest.json + cleaned_structures/).
        msa_dir: generate-msas output directory, after fetch-templates.
        cache_file: dataset_cache.json from materialize-dataset-cache; updated in place.
        min_release_date_diff: Days a template must predate each structure's release
            date; excludes the structure itself.
        max_release_date: Optional global template release-date cutoff, on top of
            min_release_date_diff.
        template_cutoff_date: Keep templates released before this date, in place of both
            release-date settings above. Leave the sequence filters on: they are then what
            keeps a structure from being its own template.
        max_templates: Templates kept per chain.
        n_processes: OF3 preprocessing worker processes.
        ccd_path: CCD for preparsing template structures; None uses Biotite's.
        sequence_filter: OF3's template sequence filters; None uses OF3's defaults.

    Returns:
        Chains with templates, chains whose hits the release-date filter removed, and
        hits the sequence filters dropped.
    """
    sequence_filter = sequence_filter or TemplateSequenceFilterSettings()
    data_dir = msa_dir / TEMPLATE_DATA_DIRNAME
    data_dir.mkdir(parents=True, exist_ok=True)
    queries = collect_msa_queries(output_dir)  # rep_id -> sequence
    representatives_fasta = data_dir / "alignment_representatives.fasta"
    with representatives_fasta.open("w") as f:
        for rep_id, seq in queries.items():
            f.write(f">{rep_id}\n{seq}\n")

    _invalidate_template_cache_on_filter_change(data_dir, sequence_filter)
    cutoff_record = data_dir / TEMPLATE_CUTOFF_FILENAME
    cutoff_record.unlink(missing_ok=True)
    if template_cutoff_date is not None:
        min_release_date_diff, max_release_date = None, template_cutoff_date
        cutoff_record.write_text(
            json.dumps({"template_cutoff_date": template_cutoff_date.date().isoformat()}) + "\n"
        )
    alignment_dir = data_dir / FILTERED_TEMPLATE_ALIGNMENTS_DIRNAME
    dropped_by_rep = _write_filtered_template_alignments(
        queries, msa_dir, alignment_dir, sequence_filter
    )
    sequence_filter_report = data_dir / SEQUENCE_FILTER_REPORT_FILENAME
    sequence_filter_report.write_text(
        "alignment_representative_id\ttemplate_id\n"
        + "".join(
            f"{rep_id}\t{template_id}\n"
            for rep_id, dropped in dropped_by_rep.items()
            for template_id in dropped
        )
    )

    settings = TemplatePreprocessorSettings(
        mode="train",
        output_directory=data_dir,
        structure_directory=msa_dir / TEMPLATE_STRUCTURES_DIRNAME,
        template_alignment_directory=alignment_dir,
        alignment_representatives_fasta=representatives_fasta,
        min_release_date_diff=min_release_date_diff,
        max_release_date=max_release_date,
        max_templates=max_templates,
        # fetch-templates already resolved every structure the .m8 files reference.
        fetch_missing_structures=False,
        # Training reads template structures as preparsed per-chain arrays.
        preparse_structures=True,
        n_processes=n_processes,
        ccd_file_path=ccd_path,
    )
    dataset_cache = read_datacache(cache_file)
    TemplatePreprocessor(input_set=dataset_cache, config=settings)()
    write_datacache_to_json(dataset_cache, cache_file)

    # OF3 filters each chain's hits against its structure's release date without logging
    # what it drops; record it so chains left without templates are visible.
    report_path = data_dir / "template_date_filter.tsv"
    rows = [
        "pdb_id\tlabel_asym_id\talignment_representative_id\trelease_date\t"
        "n_sequence_filtered\tn_hits\tn_kept"
    ]
    n_with_templates = 0
    date_filtered = []
    for pdb_id, structure in dataset_cache.structure_data.items():
        for chain in structure.chains.values():
            n_kept = len(chain.template_ids or [])
            n_with_templates += n_kept > 0
            rep_id = chain.alignment_representative_id
            if rep_id is None:
                continue
            n_hits = len(find_template_ids(rep_id, data_dir / "template_cache"))
            rows.append(
                f"{pdb_id}\t{chain.label_asym_id}\t{rep_id}\t"
                f"{str(structure.release_date)[:10]}\t"
                f"{len(dropped_by_rep.get(rep_id, []))}\t{n_hits}\t{n_kept}"
            )
            if n_hits and not n_kept:
                date_filtered.append(f"{pdb_id}/{chain.label_asym_id}")
    report_path.write_text("\n".join(rows) + "\n")
    return PreprocessTemplatesSummary(
        n_chains_with_templates=n_with_templates,
        date_filtered_chains=date_filtered,
        report_path=report_path,
        n_sequence_filtered=sum(len(dropped) for dropped in dropped_by_rep.values()),
        sequence_filter_report_path=sequence_filter_report,
    )


def _invalidate_template_cache_on_filter_change(
    data_dir: Path, settings: TemplateSequenceFilterSettings
) -> None:
    """Delete OF3's template cache if it was built with other sequence-filter settings.

    OF3 reuses existing ``template_cache/<rep_id>.npz`` entries, which would otherwise
    keep hits the new settings drop. A cache without a settings file predates the filter.

    Args:
        data_dir: ``{msa_dir}/of3_template_data``.
        settings: The settings this run uses; recorded for the next run.
    """
    settings_file = data_dir / SEQUENCE_FILTER_SETTINGS_FILENAME
    cache_dir = data_dir / "template_cache"
    previous = json.loads(settings_file.read_text()) if settings_file.exists() else None
    if cache_dir.exists() and previous != settings.model_dump():
        logger.warning(
            f"Rebuilding {cache_dir}: it was built with template sequence-filter settings "
            f"{previous or 'from before the filter existed'}, this run uses {settings.model_dump()}"
        )
        shutil.rmtree(cache_dir)
    settings_file.write_text(json.dumps(settings.model_dump(), indent=2) + "\n")


def _write_filtered_template_alignments(
    queries: dict[str, str],
    msa_dir: Path,
    dest_root: Path,
    settings: TemplateSequenceFilterSettings,
) -> dict[str, list[str]]:
    """Write each representative's sequence-filtered ``.m8`` for OF3 to read.

    Args:
        queries: {rep_id: sequence} of the dataset.
        msa_dir: generate-msas output directory, after fetch-templates.
        dest_root: Rewritten from scratch as ``{dest_root}/<rep_id>/colabfold_template.m8``.
        settings: Filter thresholds.

    Returns:
        {rep_id: template IDs dropped} for representatives with a ``.m8``.
    """
    if dest_root.exists():
        shutil.rmtree(dest_root)
    source_root = msa_dir / TEMPLATE_ALIGNMENTS_DIRNAME
    structure_dir = msa_dir / TEMPLATE_STRUCTURES_DIRNAME
    dropped_by_rep = {}
    for rep_id, seq in queries.items():
        source = source_root / rep_id / TEMPLATE_ALIGNMENT_FILENAME
        if not source.exists():
            continue
        kept_text, dropped = filter_template_alignment(
            source.read_text(), seq, structure_dir, settings
        )
        dropped_by_rep[rep_id] = dropped
        if kept_text:
            dest = dest_root / rep_id / TEMPLATE_ALIGNMENT_FILENAME
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(kept_text)
    n_dropped = sum(len(dropped) for dropped in dropped_by_rep.values())
    logger.info(
        f"Template sequence filters dropped {n_dropped} hit(s) across "
        f"{len(dropped_by_rep)} representative(s)"
    )
    return dropped_by_rep
