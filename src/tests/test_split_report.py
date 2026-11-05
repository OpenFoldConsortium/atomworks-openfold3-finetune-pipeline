"""Train/val similarity report: closest protein and ligand pairs, and the report text."""

import pytest
from rdkit import Chem

from aw_of3_finetune.split import (
    SplitResult,
    format_split_report,
    split_dataset_cache,
    split_paths,
    top_ligand_pair,
    top_protein_pair,
)
from validate_helpers import mutable_copy

METHOD_LINE = (
    "Similarity: proteins by global alignment (BLOSUM62, gap -10/-1), identity over the shorter sequence; "
    "ligands by Morgan fingerprints (radius 2, 2048 bits), Tanimoto\n"
)


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


def result_with_counts(n_train=2, n_val=2, n_test=1):
    return SplitResult(
        n_train=n_train, n_val=n_val, n_test=n_test, paths=None, warnings=[], report=""
    )


def test_identical_sequences_score_100():
    val = {("7riv", "1"): "MKTAYIAK"}
    train = {("4x1f", "1"): "MKTAYIAK", ("2crb", "1"): "GSSGSSG"}
    expected = (("7riv", "1"), ("4x1f", "1"), 100.0)
    assert top_protein_pair(val, train) == expected


def test_identity_is_over_the_shorter_sequence():
    val = {("a", "1"): "MKTAYIAK"}
    train = {("b", "1"): "MKTAYIAKQRQISFVKSHFSRQ"}  # val is an exact prefix
    _, _, actual_identity = top_protein_pair(val, train)
    assert actual_identity == 100.0


LONG_SEQUENCE = (
    "MAEVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"
    "DRLSITIRPRYYGLDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSS"
    "SLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTK"
)
UNRELATED_SHORT = "WHDQNPCIFMYLRKEGATSVWHDQNPCIFMYLRKEGATSVWHDQNPCIFMYLRKEGATSVWHDQNPCIFMY"


def test_unrelated_short_sequence_does_not_score_high():
    val = {("a", "1"): LONG_SEQUENCE}
    train = {("b", "1"): UNRELATED_SHORT}
    _, _, actual_identity = top_protein_pair(val, train)
    assert actual_identity < 50.0


def test_exact_substring_still_scores_100():
    val = {("a", "1"): LONG_SEQUENCE}
    train = {("b", "1"): LONG_SEQUENCE[100:170]}
    _, _, actual_identity = top_protein_pair(val, train)
    assert actual_identity == 100.0


def test_letters_biotite_rejects_do_not_crash():
    val = {("a", "1"): "MKTAUIAK"}
    train = {("b", "1"): "MKTAYIAK"}
    _, _, actual_identity = top_protein_pair(val, train)
    assert 80.0 <= actual_identity < 100.0


def test_protein_ties_pick_the_smallest_keys():
    val = {("b", "1"): "MKTAYIAK", ("a", "1"): "MKTAYIAK"}
    train = {("z", "1"): "MKTAYIAK", ("y", "1"): "MKTAYIAK"}
    expected = (("a", "1"), ("y", "1"), 100.0)
    assert top_protein_pair(val, train) == expected


def test_protein_pair_is_none_without_either_side():
    assert top_protein_pair({}, {("b", "1"): "MKT"}) is None
    assert top_protein_pair({("a", "1"): "MKT"}, {}) is None


def test_closest_ligand_pair_wins():
    val = {("a", "BNZ"): mol("c1ccccc1")}
    train = {("b", "TOL"): mol("Cc1ccccc1"), ("c", "ETH"): mol("CCO")}
    actual_val, actual_train, _ = top_ligand_pair(val, train)
    assert actual_val == ("a", "BNZ")
    assert actual_train == ("b", "TOL")


def test_single_atom_ligands_are_excluded():
    val = {("a", "ZN"): mol("[Zn+2]")}
    train = {("b", "ZN"): mol("[Zn+2]")}
    assert top_ligand_pair(val, train) is None


def test_report_lines_match_the_format():
    proteins = ({("7riv", "1"): "MKTAYIAK"}, {("4x1f", "1"): "MKTAYIAK"})
    ligands = ({("7riv", "BNZ"): mol("c1ccccc1")}, {("2q2k", "BNZ"): mol("c1ccccc1")})
    expected = (
        METHOD_LINE
        + "Split: 2 train, 2 val, 1 test (5 entries)\n"
        + "Most similar proteins: val 7riv chain 1 <-> train 4x1f chain 1, 100.0% identity\n"
        + "Most similar ligands:  val 7riv BNZ <-> train 2q2k BNZ, Tanimoto 1.00\n"
    )
    assert format_split_report(result_with_counts(), proteins, ligands) == expected


def test_exclusion_list_ligands_are_not_compared():
    isopropanol = "CC(C)O"
    val = {("6hj2", "IPA"): mol(isopropanol), ("6hj2", "BNZ"): mol("c1ccccc1")}
    train = {("7ax9", "IPA"): mol(isopropanol), ("7ax9", "MBN"): mol("Cc1ccccc1")}
    actual_val_key, actual_train_key, _ = top_ligand_pair(val, train)
    assert actual_val_key == ("6hj2", "BNZ")
    assert actual_train_key == ("7ax9", "MBN")


def test_report_says_none_when_val_has_only_exclusion_list_ligands():
    ligands = ({("6hj2", "IPA"): mol("CC(C)O")}, {("7ax9", "BNZ"): mol("c1ccccc1")})
    actual = format_split_report(result_with_counts(), ({}, {}), ligands)
    expected_line = "Most similar ligands:  none (no ligands in val)\n"
    assert expected_line in actual


def test_report_says_none_without_val_proteins():
    proteins = ({}, {("4x1f", "1"): "MKT"})
    actual = format_split_report(result_with_counts(), proteins, ({}, {}))
    expected_line = "Most similar proteins: none (no protein chains in val)\n"
    assert expected_line in actual


def test_report_says_none_without_ligands():
    ligands = ({("7riv", "BNZ"): mol("c1ccccc1")}, {})
    actual = format_split_report(result_with_counts(), ({}, {}), ligands)
    expected_line = "Most similar ligands:  none (no ligands in train)\n"
    assert expected_line in actual


@pytest.fixture
def built_copy(built_dataset, tmp_path):
    return mutable_copy(built_dataset, tmp_path / "copy")


def test_split_writes_the_report(built_copy, tmp_path):
    csv = tmp_path / "split.csv"
    csv.write_text("id,split\n7riv,val\n3u8v,val\n")
    result = split_dataset_cache(built_copy.output_dir, csv)
    paths = split_paths(built_copy.output_dir)
    assert paths.report.read_text() == result.report
    assert result.report.startswith("Similarity: ")
    assert "Split: 3 train, 2 val, 0 test (5 entries)\n" in result.report
    assert "Most similar proteins: val " in result.report
    assert "Most similar ligands:  " in result.report


def test_missing_reference_mol_sdf_warns(built_copy, tmp_path):
    (built_copy.output_dir / "reference_mols" / "P06.sdf").unlink()
    csv = tmp_path / "split.csv"
    csv.write_text("id,split\n7riv,val\n3u8v,val\n")
    result = split_dataset_cache(built_copy.output_dir, csv)
    expected_warning = "reference molecule P06 has no readable SDF; left out of the report"
    assert expected_warning in result.warnings

