#!/usr/bin/python3
# -*- coding: utf-8 -*-


def calc_proj_mat_size(mask, fasta_type="AA"):
    """
    Calculate the size of the projection matrix based on the mask and fasta_type.

    Args:
        mask (list): Mask values.
        fasta_type (str, optional): Type of the FASTA sequence. Defaults to 'AA'.

    Returns:
        int: Size of the projection matrix.
    """

    if fasta_type == "AA":
        x = 20
    elif fasta_type == "NT":
        x = 4
    else:
        raise ValueError("Invalid fasta_type")

    lines = (x ** mask[0]) * (x ** mask[2])
    return lines
