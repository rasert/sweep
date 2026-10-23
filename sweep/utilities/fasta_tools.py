#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
from Bio import SeqIO


def fastaread(fastaname):
    """
    Read sequences from a FASTA file.

    Args:
        fastaname (str): Path to the FASTA file.

    Returns:
        list: List of SeqRecord objects containing the sequences.
    """

    records = list(SeqIO.parse(fastaname, "fasta"))
    return records


def extract_headers_and_sequences(
    seq_records: List[SeqIO.SeqRecord],
) -> Tuple[List[str], List[str]]:
    """
    Extracts headers and sequences from a list of SeqRecord objects.

    Args:
        - seq_records (List[SeqIO.SeqRecord]): List of SeqRecord objects.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of headers (sequence IDs).
            - List of sequences (as strings).
    """
    headers = []
    sequences = []

    for record in seq_records:
        headers.append(record.id)
        sequences.append(str(record.seq))

    return headers, sequences
