-----------------------------------
SWeeP: Spaced Words Projection
-----------------------------------

This Python package implements the SWeeP (Spaced Words Projection), a
method for representing biological sequences in compact and fixed-length
feature vectors.

# Installation

To use SWeeP in Python, install the package with the following command:

    pip install sweep

# Usage

## Downloading the Default Projection Matrix

In the first use of fas2sweep with the default parameter, it will be
necessary to download the default projection matrix. It is not necessary
for use with custom projection matrix, as demonstrated in the "Changing
Projection Matrix" topic. Here is how to perform the default matrix download:
```python
from sweep import down_proj_mat

down_proj_mat()  # Downloads the default projection matrix file
```

## Handling Amino Acid Sequences

The default configurations of SWeeP are intended for vectorization of
amino acid sequences. The default output is a matrix already projected
with 600 columns. Here is an example of how to use SWeeP with amino acid
sequences:
```python
from sweep import fastaread, fas2sweep

fasta = fastaread("fasta_file_path")
vect = fas2sweep(fasta)
```

## Changing Projection Matrix

To change the projection matrix, a new orthonormal matrix can be
generated using the orthbase function. Here is an example of how
to change the projection size to 300:
```python
from sweep import fastaread, fas2sweep, orthbase

ob = orthbase(160000, 300)
fasta = fastaread("fasta_file_path")
vect = fas2sweep(fasta, orth_mat=ob)
```

## Handling Nucleotide Sequences

For nucleotide sequences, there is no default projection matrix
available in this version. Therefore, to work with nucleotides
is possible to create a custom projection matrix using the
orthbase function. The matrix size can be calculated using
the calc_proj_mat_size function. Here is an example:
```python
from sweep import fastaread, fas2sweep, orthbase, calc_proj_mat_size

mask = [4, 7, 4]
matrix_size = calc_proj_mat_size(mask, "NT")
ob = orthbase(matrix_size, 600)
fasta = fastaread("fasta_file_path")
vect = fas2sweep(fasta, mask=mask, orth_mat=ob, fasta_type="NT")
```

## Phylogenetic Tree Generation Using SWeeP

This example demonstrates the usage of the generate_tree function
from the sweep package. Initially, a mask is defined for nucleotide
sequences, and the projection matrix size is calculated accordingly.
An orthonormal projection matrix is generated using `orthbase`.
Sequences are then read from a FASTA file using `fastaread`.
By employing `fas2sweep`, these sequences are transformed into SWeeP
vectors based on the defined mask and orthonormal matrix. Lastly,
generate_tree is invoked with these vectors, sequence headers, and
a specified file name to save the resulting phylogenetic tree in
Newick format.
```python
from sweep import (
    fastaread,
    fas2sweep,
    orthbase,
    calc_proj_mat_size,
    generate_tree,
    extract_headers_and_sequences,
)

# Define the mask for nucleotide sequences
mask = [4, 7, 4]  # Example mask for nucleotide sequences

# Calculate the projection matrix size based on the mask and sequence type ('NT' for nucleotides)
matrix_size = calc_proj_mat_size(mask, "NT")

# Generate an orthonormal projection matrix using orthbase
ob = orthbase(matrix_size, 600)  # 600 is an arbitrary size example for illustration

# Read sequences from a FASTA file containing nucleotide sequences
fasta = fastaread("fasta_file_path")

# Extract headers and sequences from the FASTA file
headers, _ = extract_headers_and_sequences(fasta)

# Convert nucleotide sequences to SWeeP vectors using fas2sweep
vect = fas2sweep(fasta, mask=mask, orth_mat=ob, fasta_type="NT")

# Generate a phylogenetic tree from the SWeeP vectors and save it in Newick format
tree = generate_tree(vect, headers, "tree.newick")
```

## Available Functions

Here is a summary of the
functions available in the SWeeP package:

---

### ``fastaread``

- **Description:** Reads a FASTA file and returns a list of sequence records.
- **Input:** ``fastaname`` (str) - Path to the FASTA file.
- **Output:** ``records`` (list) - List of sequence records.

---

### ``fas2sweep``

- **Description:** Converts a list of sequences into SWeeP vectors.
- **Input:** ``fasta`` (list) - List of sequence records.
- **Output:** ``vect`` (numpy.ndarray) - SWeeP vectors.

---

### ``orthbase``

- **Description:** Generates an orthonormal projection matrix of the specified size.
- **Input:**
  - ``lin`` (int) - Number of rows.
  - ``col`` (int): Number of columns in the matrix.
  - ``seed`` (int, optional): Seed for the random number generator.
- **Output:** ``mret`` (numpy.ndarray) - Orthonormal matrix.

---

### ``calc_proj_mat_size``

- **Description:** Calculates the number of lines in the projection matrix for a given mask.
- **Input:** ``mask`` (list) - Mask specifying dimensions.
- **Output:** ``lines`` (int) - Number of lines in the matrix.

---

### ``down_proj_mat``

- **Description:** Downloads the default projection matrix file.
- **Input:** ``destination`` (str) - Path to the destination file (optional).
- **Output:** None.

---

### ``return_proj_mat_not_found_error``

- **Description:** Raises an exception indicating that the default projection matrix is not found.
- **Input:** None.
- **Output:** None.

---

### ``check_default_proj_mat``

- **Description:** Checks if the default projection matrix exists and matches the expected MD5 hash.
- **Input:** ``file`` (str) - Path to the projection matrix file.
- **Output:** None.

---

### ``get_default_proj_mat``

- **Description:** Retrieves the default projection matrix.
- **Input:** None.
- **Output:** ``orth_mat`` (numpy.ndarray) - Projection matrix.

---

### ``scipy_tree_to_newick``

- **Description:** Constructs a Newick tree from a SciPy hierarchy ClusterNode.
- **Input:**
  - ``node`` (scipy.cluster.hierarchy.ClusterNode): Root node from hierarchical clustering.
  - ``parentdist`` (float): Distance from parent node to `node`.
  - ``leaf_names`` (List[str]): Names of the leaf nodes.
- **Output:** (str) - String in Newick format.

---

### ``generate_tree``

- **Description:** Generates a Newick format tree from an input matrix.
- **Input:**
  - ``mat`` (List[List[float]]): Input matrix.
  - ``labels`` (List[str]): List of labels for the tree leaves.
  - ``output_file`` (str, optional): Name of the output file. If provided, the tree will be saved to this file.
  - ``metric`` (str): Distance metric to be used. Default is 'euclidean'.
  - ``method`` (str): Linkage method to be used in hierarchical clustering. Default is 'complete'.
- **Output:** (str) - String in Newick format representing the tree.

---

### `extract_headers_and_sequences`

- **Description:** Extracts headers and sequences from a list of SeqRecord objects.
- **Input:**
  - `seq_records` (List[SeqIO.SeqRecord]): List of SeqRecord objects.
- **Output:** 
  - (Tuple[List[str], List[str]]) - A tuple containing two lists:
    - List of headers (sequence IDs).
    - List of sequences (as strings).

---

## Article Reference

If you use the SWeeP algorithm or this Python package in your work, please cite the
following article:

```bib
@article{Pierri2020,
  title={SWeeP: representing large biological sequences datasets in compact vectors},
  author={De Pierri, Camilla Reginatto and Voyceik, Ricardo and Santos de Mattos, Letícia Graziela Costa and Kulik, Mariane Gonçalves and Camargo, Josué Oliveira and Repula de Oliveira, Aryel Marlus and de Lima Nichio, Bruno Thiago and Marchaukoski, Jeroniza Nunes and da Silva Filho, Antonio Camilo and Guizelini, Dieval and Ortega, J. Miguel and Pedrosa, Fabio O. and Raittz, Roberto Tadeu},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={91},
  year={2020},
  doi={10.1038/s41598-019-55627-4},
  url={https://doi.org/10.1038/s41598-019-55627-4},
  issn={2045-2322}
}
```
