#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .sweep_support import mask2vec_bin, mask2vec_count, generate_chunk
from .utilities import fastaread
from .default_proj_mat_ope import check_default_proj_mat
import math
import h5py
import os
import numpy as np
import sys
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings


def fas2sweep(
    xfas,
    orth_mat=None,
    mask=None,
    composition="binary",
    verbose=False,
    verbose_start="",
    chunk_size=1000,
    projection=True,
    fasta_type="AA",
    skip_seq_len_check=False,
    dtype=None,
    n_jobs=1,
):
    warnings.warn(
        "'fas2sweep' is deprecated and will be removed in a future release; use 'sweep' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
    Perform the SWeeP algorithm on sequences in FASTA format.

    Args:
        - xfas (str or list): Path to the FASTA file or a list of FASTA
          sequences.
        - orth_mat (ndarray, optional): Projection matrix. If not provided,
          the default matrix will be used.
        - mask (ndarray, optional): Mask for encoding. If not provided, the
          default mask [2, 1, 2] will be used.
        - composition (str, optional): Composition type. Either 'binary' or
          'count'. Defaults to 'binary'.
        - verbose (bool, optional): Verbosity mode. Defaults to False.
        - chunk_size (int, optional): Size of each chunk for concurrent
          processing by job. Defaults to 1000.
        - projection (bool, optional): Perform projection. Defaults to True.
        - fasta_type (str, optional): Type of FASTA sequences. Either 'AA' for
          amino acids or 'NT' for nucleotides. Defaults to 'AA'.
        - skip_seq_len_check (bool, optional): Skip the check for sequence
          length. Defaults to False.
        - dtype (dtype, optional): Data type of the output matrix. If not
          provided, float32 is used for projection and int32 for no projection.
        - n_jobs (int, optional): Number of parallel jobs. Defaults to 1.

    Returns:
        - ndarray or sparse matrix: Resulting SWeeP matrix.
    """

    if dtype is None:
        if projection:
            dtype = np.float32
        else:
            dtype = np.int32

    if mask is None:
        mask = [2, 1, 2]
    mask = np.array(mask)

    if fasta_type == "AA":
        defSize = 20
    elif fasta_type == "NT":
        defSize = 4

    mask_sum = sum([mask[0], mask[2]])

    # Check if the mask is valid
    if len(mask) != 3 or not (isinstance(sum(mask), np.integer)):
        message = "Mask must be an array with 3 integer values."
        raise Exception(message)

    # Check if orth_mat is unnecessary when projection is disabled
    elif not (orth_mat is None) and not projection:
        raise Exception("The orth_mat parameter is unnecessary if " + "projection=False.")

    # Check if the size of mask parts is too high
    elif (mask_sum > 5 and fasta_type == "AA") or (mask_sum > 10 and fasta_type == "NT"):
        raise Exception("The size of the mask parts is too high.")

    # Extract sequences from the FASTA file
    if isinstance(xfas, str):
        fas_cell = fastaread(xfas)
    else:
        fas_cell = xfas

    seqs = []
    for i in fas_cell:
        seqs.append(str(i.seq))

    # Calculate the number of chunks
    chunks = math.ceil(len(seqs) / chunk_size)
    len_seqs = len(seqs)

    # Checking if all sequences are bigger than the mask size
    if not skip_seq_len_check:
        sum_mask = sum(mask)
        for i, n in enumerate(seqs):
            if len(n) < sum_mask:
                message = "Sequence %i smaller than the mask size." % i
                raise Exception(message)

    # Generate chunk indices
    idx = generate_chunk(chunks, len_seqs) - 1
    proj_mat_size_req = defSize ** mask[0] * defSize ** mask[2]

    if projection:
        if orth_mat is None:
            if proj_mat_size_req != 160000:
                message = (
                    "The default matrix is intended for the sweep of"
                    " amino acids with the default mask, for other "
                    "cases you can disable the projection or set the "
                    "orth_mat parameter."
                )
                raise Exception(message)

            # Download default projection matrix if not available
            libLocal = os.path.dirname(os.path.realpath(__file__))
            mat_file_local = os.path.join(libLocal, "sweep-default-projection-matrix-600.mat")
            check_default_proj_mat(mat_file_local)
            orth_mat = h5py.File(mat_file_local, "r")
            var_name = list(orth_mat.keys())[0]
            orth_mat = orth_mat[var_name][()].T
        else:
            if orth_mat.shape[0] != proj_mat_size_req:
                message = (
                    "The defined orth_mat does not have the appropriate"
                    " dimensions."
                    "\n\nThe number of lines must be:"
                    "\n(x**mask[0])*(x**mask[2]),"
                    "\nwhere x=20, if fasta_type=='AA',"
                    "\nand x=4, if fasta_type=='NT'."
                    "\n\nUsing the function sweep.calc_proj_mat_size is"
                    " possible to check the necessary size and "
                    "sweep.orthbase to create the projection mat."
                    "\n\nWith the current input, the number of lines "
                    "should be {}.".format(proj_mat_size_req)
                )
                raise Exception(message)
        orth_mat = orth_mat.astype(dtype)
        m = orth_mat.shape[1]

    if composition == "binary":
        mask2vec = mask2vec_bin
    elif composition == "count":
        mask2vec = mask2vec_count

    # Define mask2vec function for conversion
    m2v = lambda a: mask2vec(a, mask, defSize)[0]

    def sweep_chunk(i):
        parcM = seqs[int(idx[i, 0]) : int(idx[i, 1]) + 1]
        out_mat = np.array([m2v(x) for x in parcM], dtype=dtype)
        if projection:
            n = out_mat.shape[0]
            out = np.zeros((n, m), dtype=dtype)
            out_mat = np.dot(out_mat, orth_mat, out=out)
        else:
            out_mat = lil_matrix(out_mat, dtype=dtype)
        return out_mat

    range_s = range(0, chunks)

    # Run sweep on chunks in parallel
    result_mat = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(sweep_chunk)(i)
        for i in tqdm(
            range_s,
            position=0,
            leave=True,
            desc=f"{verbose_start}Running SWeeP",
            file=sys.stdout,
            disable=(not verbose),
        )
    )

    # Concatenate the resulting matrices
    if projection:
        result_mat = np.vstack(result_mat)
    else:
        result_mat = lil_matrix(vstack(result_mat))

    return result_mat
