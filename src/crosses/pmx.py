import numpy as np
from numba import jit


@jit(nopython=True)
def pmx(gene1: np.ndarray, gene2: np.ndarray) -> np.ndarray:
    num_genes = gene1.shape[0]
    new_gene1 = np.copy(gene2)

    copy_start_index, copy_end_index = np.random.choice(num_genes, 2, replace=False)
    if copy_start_index > copy_end_index:
        copy_start_index, copy_end_index = copy_end_index, copy_start_index

    mapping_b_to_a = {
        gene1: gene2
        for gene2, gene1 in zip(
            gene1[copy_start_index : copy_end_index + 1],
            gene2[copy_start_index : copy_end_index + 1],
        )
    }

    for index in range(0, copy_start_index):
        _pmx_fill_non_mapped_gene(gene1, mapping_b_to_a, new_gene1, index)

    for index in range(copy_end_index + 1, num_genes):
        _pmx_fill_non_mapped_gene(gene1, mapping_b_to_a, new_gene1, index)

    return new_gene1


@jit(nopython=True)
def _pmx_fill_non_mapped_gene(
    gene: np.ndarray,
    mapping_b_to_a: dict[int, int],
    new_gene: np.ndarray,
    index: int,
) -> None:
    current_gene = gene[index]
    while current_gene in mapping_b_to_a:
        current_gene = mapping_b_to_a[current_gene]
    new_gene[index] = current_gene
