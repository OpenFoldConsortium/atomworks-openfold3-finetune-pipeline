"""Turn one in-house structure (.pdb or .cif plus a ligand table) into a pipeline-ready mmCIF.

Each listed ligand gets a non-CCD residue name and an inline chem_comp_atom/chem_comp_bond
definition with the SMILES' bond orders, so finetune-prep never looks it up in the CCD.
"""

import contextlib
import dataclasses
import io
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import atomworks.io as aw_io
import numpy as np
from atomworks.io.config import ParseConfig, PrepareConfig
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.ccd import chem_comp_to_one_letter, custom_ccd_residues
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_buffer
from atomworks.io.utils.sequence import aa_chem_comp_1to3
from biotite.structure import (
    AtomArray,
    AtomArrayStack,
    BondType,
    get_residue_starts,
    residue_iter,
)
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile

from aw_of3_finetune.ligand_inputs import (
    CcdCodeCollisionError,
    LigandNamer,
    LigandRow,
    _definition_key,
    ccd_codes,
    ligand_definition,
    read_restraint_bonds,
    unlisted_placeholders,
)
from aw_of3_finetune.constants import PREPARE_RECORD_FILE
from aw_of3_finetune.parsing import STANDARD_PARSE_KWARGS
from aw_of3_finetune.pipeline import _read_release_date, _read_resolution

DEFAULT_RELEASE_DATE = "2026-01-01"
DEFAULT_METHOD = "X-RAY DIFFRACTION"
PREPARE_DROPPED_FILE = "prepare_dropped.json"
LIGANDS_FILE = "ligands.json"
INPUT_SUFFIXES = (".pdb", ".cif")
_AROMATIC_BONDS = (BondType.AROMATIC_SINGLE, BondType.AROMATIC_DOUBLE)
_WATERS = ("HOH", "DOD", "WAT")
# Missing atoms are finetune-prep's to add: it does for any input, and they have no coordinates.
_PARSE_KWARGS = {**STANDARD_PARSE_KWARGS, "add_missing_atoms": False}
_PREPARE_FIELDS = {field.name for field in dataclasses.fields(PrepareConfig)}
_LOADER_LOGGER = "atomworks.io._loaders"
_LOADER_CHAIN_SPLIT_MESSAGE = "contains both polymer and non-polymer residues"
_REMARK_2_RESOLUTION = re.compile(r"^REMARK   2 RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS")
_REMARK_3_RESOLUTION = re.compile(
    r"^REMARK   3\s+RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([0-9.]+)"
)


class PreparedStructure(NamedTuple):
    """One structure's pipeline-ready mmCIF and what was decided for it."""

    cif_text: str
    ligand_codes: dict[str, str]  # input res_name -> code
    release_date_defaulted: bool


class PrepareInputsUsageError(ValueError):
    """The inputs and the ligand table disagree in a way the user must fix before any run."""


class _Header(NamedTuple):
    """Entry-level facts read from the input file."""

    release_date: str | None
    resolution: float | None
    method: str | None


def _read_input(path: Path) -> tuple[AtomArray, _Header]:
    """Model 1 (first altloc, hydrogens kept) and header facts, read without CCD lookups."""
    if path.suffix.lower() == ".pdb":
        atoms = PDBFile.read(path).get_structure(model=1, altloc="first")
        return atoms, _pdb_header(path)
    cif = CIFFile.read(path)
    atoms = pdbx.get_structure(cif, model=1, altloc="first")
    block = cif.block
    has_date = (
        "pdbx_audit_revision_history" in block
        and "revision_date" in block["pdbx_audit_revision_history"]
    )
    release_date = _read_release_date(block).isoformat() if has_date else None
    method = block["exptl"]["method"].as_array()[0] if "exptl" in block else None
    return atoms, _Header(release_date, _read_resolution(block), method)


def _pdb_header(path: Path) -> _Header:
    """Resolution (REMARK 2, else REMARK 3) and method (EXPDTA) from a .pdb header."""
    remark_2 = remark_3 = method = None
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            break
        if match := _REMARK_2_RESOLUTION.match(line):
            remark_2 = float(match.group(1))
        elif match := _REMARK_3_RESOLUTION.match(line):
            remark_3 = float(match.group(1))
        elif line.startswith("EXPDTA"):
            method = line[10:].strip()
    resolution = remark_2 if remark_2 is not None else remark_3
    return _Header(None, resolution, method)


def _row_definition(atoms: AtomArray, row: LigandRow) -> AtomArray:
    """The ligand definition for row, checked to be the same for every copy in the structure.

    Args:
        atoms: The input's atoms, hydrogens included.
        row: One ligand-table row of this structure.

    Returns:
        The first copy's definition (output of ligand_definition).

    Raises:
        ValueError: The residue is absent, or two copies give different definitions.
        LigandMatchError: The SMILES does not match a copy.
    """
    copies = atoms[atoms.res_name == row.res_name]
    if len(copies) == 0:
        raise ValueError(f"ligand table row {row.res_name!r}: no residue with that name")
    restraint = (
        read_restraint_bonds(row.restraint_cif, row.res_name) if row.restraint_cif else None
    )
    definitions = [
        ligand_definition(
            row.smiles, list(copy.atom_name), list(copy.element), copy.coord, restraint
        )
        for copy in residue_iter(copies)
    ]
    first_key = _definition_key(definitions[0], row.smiles)
    if any(_definition_key(other, row.smiles) != first_key for other in definitions[1:]):
        raise ValueError(f"{row.res_name!r}: copies in this structure differ in atoms or bonds")
    return definitions[0]


def _registry_entry(definition: AtomArray, res_name: str) -> AtomArray:
    """The definition with CCD-template annotations; ring bonds as plain Kekulé orders.

    OF3's mol_from_atomarray reads aromatic-flagged bonds as RDKit AROMATIC and then cannot
    kekulize an [nH] ring, so bonds carry no aromatic flag; RDKit re-perceives aromaticity.
    """
    entry = definition.copy()
    n_atoms = entry.array_length()
    entry.res_name[:] = res_name
    aromatic = np.zeros(n_atoms, dtype=bool)
    for i, j, bond_type in entry.bonds.as_array():
        if BondType(bond_type) in _AROMATIC_BONDS:
            aromatic[[i, j]] = True
    entry.bonds.remove_aromaticity()
    entry.set_annotation("chem_comp_type", np.full(n_atoms, "NON-POLYMER"))
    entry.set_annotation("is_aromatic", aromatic)
    entry.set_annotation("is_leaving_atom", np.zeros(n_atoms, dtype=bool))
    return entry


class _DropLoaderChainSplit(logging.Filter):
    """Drops atomworks' "separating them ... naming the non-polymer residues as B" warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _LOADER_CHAIN_SPLIT_MESSAGE not in record.getMessage()


@contextlib.contextmanager
def _quiet_loader_chain_split() -> Iterator[None]:
    """Silence that one loader warning; _one_chain_per_hetero_residue replaces its split."""
    logger = logging.getLogger(_LOADER_LOGGER)
    log_filter = _DropLoaderChainSplit()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


def _one_chain_per_hetero_residue(
    atoms: AtomArrayStack, author_chain_ids: np.ndarray
) -> tuple[AtomArrayStack, dict[str, str]]:
    """Give each non-water hetero residue its own chain, as label_asym_id does in an mmCIF.

    atomworks' PDB loader puts all of a chain's hetero residues on one new chain, so two
    ligand copies would read as one two-residue ligand. Waters keep their chain.

    Args:
        atoms: load_pdb's atoms.
        author_chain_ids: The file's chain id per atom, same order as atoms.

    Returns:
        The atoms with new chain ids, and {new chain id: author chain id}.
    """
    if len(author_chain_ids) != atoms.array_length():
        raise ValueError("atomworks' PDB loader and biotite read different atoms")
    atoms = atoms.copy()
    new_ids = create_chain_id_generator(unavailable_chain_ids=np.unique(atoms.chain_id))
    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    author_by_chain = {}
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        if atoms.hetero[start] and atoms.res_name[start] not in _WATERS:
            atoms.chain_id[start:stop] = next(new_ids)
        author_by_chain[str(atoms.chain_id[start])] = str(author_chain_ids[start])
    return atoms, author_by_chain


def _cif_author_chains(path: Path) -> dict[str, str]:
    """{label_asym_id: auth_asym_id} from an mmCIF input's atom_site."""
    atom_site = CIFFile.read(path).block["atom_site"]
    if "auth_asym_id" not in atom_site:
        return {}
    pairs = zip(
        atom_site["label_asym_id"].as_array(), atom_site["auth_asym_id"].as_array(), strict=True
    )
    return {str(label): str(auth) for label, auth in pairs}


def _parse(
    path: Path, entries: dict[str, AtomArray], author_chain_ids: np.ndarray
) -> tuple[AtomArray, dict[str, str]]:
    """atomworks' parse with the pipeline's settings, the ligand entries overriding the CCD.

    Args:
        path: Input .pdb or .cif.
        entries: Registry entries by input residue name.
        author_chain_ids: The input's chain id per atom (biotite read, model 1, first altloc).

    Returns:
        The asym unit, and {chain id: author chain id}.
    """
    with custom_ccd_residues(entries):
        if path.suffix.lower() == ".pdb":
            with _quiet_loader_chain_split():
                loaded, _ = aw_io.load_pdb(_with_increasing_serials(path), model=1)
            split, author_by_chain = _one_chain_per_hetero_residue(loaded, author_chain_ids)
            kwargs = {k: v for k, v in _PARSE_KWARGS.items() if k in _PREPARE_FIELDS}
            parsed = aw_io.parse_atom_array(split, config=PrepareConfig(**kwargs))
        else:
            author_by_chain = _cif_author_chains(path)
            parsed = aw_io.parse(str(path), config=ParseConfig(**_PARSE_KWARGS))
    return parsed["asym_unit"][0], author_by_chain


def _with_increasing_serials(path: Path) -> io.StringIO:
    """The PDB file with atoms numbered in file order, which biotite 1.6's bond reader requires.

    Refinement outputs can list hetero residues out of serial order (x00242-1's raw file).

    Args:
        path: Input .pdb.

    Returns:
        The text with atom serials renumbered, CONECT records following them.
    """
    lines = path.read_text().splitlines()
    atom_lines = [i for i, line in enumerate(lines) if line.startswith(("ATOM  ", "HETATM"))]
    new_serials = {}
    for new_serial, i in enumerate(atom_lines, start=1):
        new_serials.setdefault(lines[i][6:11].strip(), new_serial)
        lines[i] = f"{lines[i][:6]}{new_serial:>5}{lines[i][11:]}"
    for i, line in enumerate(lines):
        if line.startswith("CONECT"):
            fields = [line[start:start + 5].strip() for start in range(6, len(line), 5)]
            kept = [new_serials[field] for field in fields if field in new_serials]
            lines[i] = "CONECT" + "".join(f"{serial:>5}" for serial in kept)
    return io.StringIO("\n".join(lines) + "\n")


def _rename_in_definition_order(
    atoms: AtomArray, res_name: str, code: str, names: list[str]
) -> AtomArray:
    """Rename res_name residues to code and order their atoms as in the definition."""
    rank = {name: i for i, name in enumerate(names)}
    order = np.arange(atoms.array_length())
    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        if atoms.res_name[start] == res_name:
            unknown = [str(n) for n in atoms.atom_name[start:stop] if str(n) not in rank]
            if unknown:
                raise ValueError(
                    f"{res_name} {atoms.chain_id[start]} {atoms.res_id[start]}: atom(s) "
                    f"{unknown} are not in the ligand definition"
                )
            ranks = [rank[str(name)] for name in atoms.atom_name[start:stop]]
            order[start:stop] = start + np.argsort(ranks, kind="stable")
    atoms = atoms[order]
    atoms.res_name[atoms.res_name == res_name] = code
    return atoms


def _segments(author_numbers: list[int]) -> list[tuple[int, int]]:
    """(start, stop) index ranges of the runs of consecutive author numbers."""
    breaks = [i for i in range(1, len(author_numbers)) if author_numbers[i] - author_numbers[i - 1] > 1]
    edges = [0, *breaks, len(author_numbers)]
    return list(zip(edges[:-1], edges[1:]))


def _construct_positions(
    letters: str, author_numbers: list[int], construct: str
) -> list[int] | None:
    """0-based construct position of each residue, the runs of author numbers matching exactly and in order.

    Where a run fits in several places, the one continuing the numbering of the longest run wins.

    Args:
        letters: One letter per observed residue.
        author_numbers: The residues' author numbers.
        construct: The whole sequence.

    Returns:
        One position per residue, or None if the residues do not fit.
    """
    segments = _segments(author_numbers)
    dead_ends: set[tuple[int, int]] = set()

    def place(index: int, earliest: int, shift: int) -> list[int] | None:
        if index == len(segments):
            return []
        if (index, earliest) in dead_ends:
            return None
        start, stop = segments[index]
        run = letters[start:stop]
        hits = [hit for hit in range(earliest, len(construct)) if construct.startswith(run, hit)]
        for hit in sorted(hits, key=lambda hit: abs(hit - author_numbers[start] - shift)):
            rest = place(index + 1, hit + len(run), hit - author_numbers[start])
            if rest is not None:
                return [*range(hit, hit + len(run)), *rest]
        dead_ends.add((index, earliest))
        return None

    longest_start, longest_stop = max(segments, key=lambda segment: segment[1] - segment[0])
    anchor = construct.find(letters[longest_start:longest_stop])
    return place(0, 0, anchor - author_numbers[longest_start]) if anchor >= 0 else None


def _chain_positions(
    chain: str, letters: str, author_numbers: list[int], constructs: tuple[str, ...]
) -> tuple[list[int], str | None]:
    """Where one polymer chain's residues sit: in the first construct that fits, else in the observed residues.

    Args:
        chain: Chain id, for the error.
        letters: One letter per observed residue.
        author_numbers: The residues' author numbers.
        constructs: The candidate whole sequences.

    Returns:
        0-based positions, and the construct they index (None when numbered over the observed residues).

    Raises:
        ValueError: The author numbers have gaps and no construct fits. The sequence would then
            join residues that are not neighbours, and OF3's CA-gap filter would drop the chain.
    """
    for construct in constructs:
        positions = _construct_positions(letters, author_numbers, construct)
        if positions is not None:
            return positions, construct
    if len(_segments(author_numbers)) > 1:
        raise ValueError(
            f"chain {chain} has gaps in its residue numbers and no construct fits it; "
            "give its whole sequence with --constructs"
        )
    return list(range(len(letters))), None


def _number_polymers(
    atoms: AtomArray, constructs: tuple[str, ...], ligand_codes: set[str]
) -> tuple[dict[tuple[str, int], int], dict[str, list[str]]]:
    """Set each polymer residue's number to its 1-based sequence position, in place.

    Args:
        atoms: The parsed structure.
        constructs: The candidate whole sequences.
        ligand_codes: Listed ligands; a .cif may label one as part of the polymer's chain.

    Returns:
        {(chain id, new number): author number}, and {chain id: residue names of the whole
        sequence} for the chains placed in a construct.
    """
    has_author_ids = "auth_seq_id" in atoms.get_annotation_categories()
    author_ids = atoms.auth_seq_id.copy() if has_author_ids else atoms.res_id.copy()
    starts = get_residue_starts(atoms, add_exclusive_stop=True)
    residues: dict[str, list[tuple[int, int]]] = {}
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        if atoms.is_polymer[start] and str(atoms.res_name[start]) not in ligand_codes:
            residues.setdefault(str(atoms.chain_id[start]), []).append((int(start), int(stop)))
    to_letter, to_name = chem_comp_to_one_letter(), aa_chem_comp_1to3()
    author_numbers: dict[tuple[str, int], int] = {}
    sequences: dict[str, list[str]] = {}
    for chain, ranges in residues.items():
        names = [str(atoms.res_name[start]) for start, _ in ranges]
        letters = "".join(to_letter.get(name, "X") for name in names)
        numbers = [int(author_ids[start]) for start, _ in ranges]
        positions, construct = _chain_positions(chain, letters, numbers, constructs)
        for (start, stop), position, number in zip(ranges, positions, numbers, strict=True):
            atoms.res_id[start:stop] = position + 1
            author_numbers[chain, position + 1] = number
        if construct is not None:
            sequence = [to_name.get(letter, "UNK") for letter in construct]
            for position, name in zip(positions, names, strict=True):
                sequence[position] = name
            sequences[chain] = sequence
    return author_numbers, sequences


def _write_sequences(block: pdbx.CIFBlock, sequences: dict[str, list[str]]) -> None:
    """Put the whole sequences in entity_poly and entity_poly_seq (the writer lists observed residues).

    Args:
        block: The written structure.
        sequences: {chain id: residue names of the whole sequence}.
    """
    atom_site = block["atom_site"]
    entity_by_chain = dict(
        zip(atom_site["label_asym_id"].as_array(str), atom_site["label_entity_id"].as_array(str))
    )
    by_entity = {entity_by_chain[chain]: names for chain, names in sequences.items()}
    entity_poly, poly_seq = block["entity_poly"], block["entity_poly_seq"]
    to_letter = chem_comp_to_one_letter()
    can = entity_poly["pdbx_seq_one_letter_code_can"].as_array(str).tolist()
    code = entity_poly["pdbx_seq_one_letter_code"].as_array(str).tolist()
    for row, entity in enumerate(entity_poly["entity_id"].as_array(str)):
        if entity in by_entity:
            names = by_entity[entity]
            can[row] = "".join(to_letter.get(name, "X") for name in names)
            code[row] = "".join(
                to_letter[name] if name in STANDARD_AA else f"({name})" for name in names
            )
    block["entity_poly"]["pdbx_seq_one_letter_code_can"] = can
    block["entity_poly"]["pdbx_seq_one_letter_code"] = code

    rows = [
        (entity, hetero, name, number)
        for entity, hetero, name, number in zip(
            *(poly_seq[column].as_array(str) for column in ("entity_id", "hetero", "mon_id", "num")),
            strict=True,
        )
        if entity not in by_entity
    ]
    for entity, names in by_entity.items():
        rows += [(entity, "n", name, str(i)) for i, name in enumerate(names, start=1)]
    rows.sort(key=lambda row: (row[0], int(row[3])))
    block["entity_poly_seq"] = pdbx.CIFCategory(
        {
            column: [row[i] for row in rows]
            for i, column in enumerate(("entity_id", "hetero", "mon_id", "num"))
        }
    )


def _keep_rows(category: pdbx.CIFCategory, keep: np.ndarray) -> pdbx.CIFCategory:
    return pdbx.CIFCategory({name: category[name].as_array()[keep] for name in category})


def _restore_author_chains(block: pdbx.CIFBlock, author_by_chain: dict[str, str]) -> None:
    """Write the input's chain ids to atom_site.auth_asym_id (label_asym_id stays per chain)."""
    atom_site = block["atom_site"]
    labels = atom_site["label_asym_id"].as_array()
    atom_site["auth_asym_id"] = [author_by_chain.get(str(label), str(label)) for label in labels]


def _restore_author_numbers(
    block: pdbx.CIFBlock, author_numbers: dict[tuple[str, int], int]
) -> None:
    """Write the polymers' author numbers to atom_site.auth_seq_id (the writer copies label_seq_id)."""
    atom_site = block["atom_site"]
    labels = atom_site["label_asym_id"].as_array(str)
    numbers = atom_site["label_seq_id"].as_array(int, -1)
    written = atom_site["auth_seq_id"].as_array(str)
    atom_site["auth_seq_id"] = [
        str(author_numbers.get((label, int(number)), author))
        for label, number, author in zip(labels, numbers, written, strict=True)
    ]


def _finish_block(
    block: pdbx.CIFBlock, codes: set[str], release_date: str, header: _Header
) -> None:
    """Keep only the custom ligands' inline definitions; add the entry-level categories."""
    for name in ("chem_comp_atom", "chem_comp_bond"):
        if name not in block:
            continue
        keep = np.isin(block[name]["comp_id"].as_array(), list(codes))
        if keep.any():
            block[name] = _keep_rows(block[name], keep)
        else:
            del block[name]
    atom_site = block["atom_site"]
    asym_to_entity = dict(
        zip(atom_site["label_asym_id"].as_array(), atom_site["label_entity_id"].as_array())
    )
    block["struct_asym"] = pdbx.CIFCategory(
        {"id": list(asym_to_entity), "entity_id": list(asym_to_entity.values())}
    )
    block["pdbx_audit_revision_history"] = pdbx.CIFCategory(
        {
            "ordinal": ["1"],
            "data_content_type": ["Structure model"],
            "major_revision": ["1"],
            "minor_revision": ["0"],
            "revision_date": [release_date],
        }
    )
    block["exptl"] = pdbx.CIFCategory(
        {"entry_id": [block.name], "method": [header.method or DEFAULT_METHOD]}
    )
    if header.resolution is not None:
        block["refine"] = pdbx.CIFCategory(
            {
                "entry_id": [block.name],
                "pdbx_refine_id": [header.method or DEFAULT_METHOD],
                "ls_d_res_high": [str(header.resolution)],
            }
        )


def prepare_structure(
    path: Path,
    rows: list[LigandRow],
    namer: LigandNamer,
    ccd: CIFFile,
    release_date: str,
    constructs: tuple[str, ...] = (),
) -> PreparedStructure:
    """Write one structure as an mmCIF whose listed ligands carry inline, non-CCD definitions.

    The structure id is the file stem; it is recorded with namer for each ligand code.

    Args:
        path: Input .pdb or .cif.
        rows: This structure's ligand-table rows.
        namer: Gives each ligand its code; shared across structures.
        ccd: Parsed CCD, for the placeholder guard.
        release_date: YYYY-MM-DD used when the input has no revision date.
        constructs: Whole sequences a polymer chain may be placed in.

    Returns:
        The CIF text, input res_name -> code, and whether release_date was used.

    Raises:
        LigandMatchError: A SMILES does not match its ligand's atoms.
        ValueError: A placeholder ligand is not listed, a row's residue is absent, copies of
            a ligand differ, or a chain has gaps and no construct fits it.
    """
    structure_id = path.stem
    atoms, header = _read_input(path)
    listed = {row.res_name for row in rows}
    unlisted = unlisted_placeholders(atoms, listed, ccd)
    if unlisted:
        raise ValueError(f"ligand residue(s) {unlisted} are not in the ligand table")
    definitions = {row.res_name: (row, _row_definition(atoms, row)) for row in rows}

    parse_entries = {name: _registry_entry(d, name) for name, (_, d) in definitions.items()}
    structure, author_by_chain = _parse(path, parse_entries, atoms.chain_id)

    ligand_codes, write_entries = {}, {}
    for name, (row, definition) in definitions.items():
        code = namer.name_for(definition, row.smiles)
        ligand_codes[name] = code
        write_entries[code] = _registry_entry(definition, code)
        names = [str(atom_name) for atom_name in definition.atom_name]
        structure = _rename_in_definition_order(structure, name, code, names)

    author_numbers, sequences = _number_polymers(structure, constructs, set(write_entries))
    config = CIFWriteConfig(
        id=structure_id,
        include_entity_categories=True,
        # include_chem_comp=True emits chem_comp_atom/chem_comp_bond (finetune-prep's writer
        # turns it off); the registry entries make the ligands' rows come from the definition.
        include_chem_comp=True,
        ccd_entries=write_entries,
        chain_disambiguation="chain_iid",
        warn_on_ccd_without_registry=False,
    )
    cif = CIFFile.read(io.StringIO(to_cif_buffer(structure, config=config).getvalue()))
    release_date_defaulted = header.release_date is None
    _finish_block(cif.block, set(write_entries), header.release_date or release_date, header)
    _restore_author_chains(cif.block, author_by_chain)
    _restore_author_numbers(cif.block, author_numbers)
    _write_sequences(cif.block, sequences)

    for code in ligand_codes.values():
        namer.add_structure(code, structure_id)
    out = io.StringIO()
    cif.write(out)
    return PreparedStructure(out.getvalue(), ligand_codes, release_date_defaulted)


def read_constructs(path: Path) -> tuple[str, ...]:
    """The sequences of a FASTA file, upper-cased.

    Args:
        path: FASTA file; a sequence may span several lines.

    Returns:
        One sequence per record, in file order.

    Raises:
        PrepareInputsUsageError: The file has no sequence.
    """
    records = re.split(r"^>.*$", path.read_text(), flags=re.MULTILINE)[1:]
    sequences = tuple("".join(record.split()).upper() for record in records)
    if not any(sequences):
        raise PrepareInputsUsageError(f"{path} has no FASTA sequence")
    return sequences


def _input_files(input_dir: Path) -> dict[str, Path]:
    """Input .pdb and .cif files by stem; a stem present twice is a usage error."""
    by_stem: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in INPUT_SUFFIXES:
            continue
        if path.stem in by_stem:
            raise PrepareInputsUsageError(
                f"{path.stem!r} is present as both {by_stem[path.stem].name} and {path.name}"
            )
        by_stem[path.stem] = path
    return by_stem


def prepare_directory(
    input_dir: Path,
    rows: list[LigandRow],
    output_dir: Path,
    ccd: CIFFile,
    release_date: str,
    constructs: tuple[str, ...] = (),
) -> dict:
    """Prepare every .pdb/.cif in input_dir; a structure that fails is dropped with its reason.

    Writes <output_dir>/<stem>.cif, prepare_dropped.json ({stem: reason}), ligands.json and
    prepare_inputs.json. A dropped structure's CIF from an earlier run is deleted.

    Args:
        input_dir: Directory of .pdb and .cif files; the file stem is the structure id.
        rows: The ligand table.
        output_dir: Created if missing.
        ccd: Parsed CCD.
        release_date: YYYY-MM-DD used for inputs without a revision date.
        constructs: Whole sequences a polymer chain may be placed in.

    Returns:
        The prepare_inputs.json content: release_date, release_date_defaulted, inputs (the
        prepared stems).

    Raises:
        PrepareInputsUsageError: A table structure_id has no input file, or a stem is both
            .pdb and .cif.
        CcdCodeCollisionError: A generated ligand code is a CCD component id (whole run).
    """
    files = _input_files(input_dir)
    unknown = sorted({row.structure_id for row in rows} - set(files))
    if unknown:
        raise PrepareInputsUsageError(
            f"ligand table structure_id(s) with no input file in {input_dir}: {unknown}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    namer = LigandNamer(ccd_codes(ccd))
    dropped: dict[str, str] = {}
    prepared: list[str] = []
    defaulted: list[str] = []
    for stem, path in files.items():
        stem_rows = [row for row in rows if row.structure_id == stem]
        try:
            result = prepare_structure(path, stem_rows, namer, ccd, release_date, constructs)
        except CcdCodeCollisionError:
            raise
        except Exception as error:  # noqa: BLE001 - one bad structure must not stop the others
            lines = str(error).splitlines()
            dropped[stem] = f"{type(error).__name__}: {lines[0] if lines else ''}".rstrip(": ")
            (output_dir / f"{stem}.cif").unlink(missing_ok=True)  # an earlier run's CIF would be ingested
            continue
        (output_dir / f"{stem}.cif").write_text(result.cif_text)
        prepared.append(stem)
        if result.release_date_defaulted:
            defaulted.append(stem)
    record = {
        "release_date": release_date,
        "release_date_defaulted": defaulted,
        "inputs": prepared,
    }
    (output_dir / PREPARE_DROPPED_FILE).write_text(json.dumps(dropped, indent=2))
    used_ligands = {code: info for code, info in namer.records().items() if info["structures"]}
    (output_dir / LIGANDS_FILE).write_text(json.dumps(used_ligands, indent=2))
    (output_dir / PREPARE_RECORD_FILE).write_text(json.dumps(record, indent=2))
    return record
