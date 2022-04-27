import numpy as np
from numba import jit


@jit(nopython=True)
def ox(
    gene1: np.ndarray, gene2: np.ndarray, returns1: np.ndarray, returns2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_genes = gene1.shape[0]
    copy_start_index, copy_end_index = np.random.choice(num_genes, 2, replace=False)
    if copy_start_index > copy_end_index:
        copy_start_index, copy_end_index = copy_end_index, copy_start_index

    new_gene = np.copy(gene1)
    new_returns = np.copy(returns1)
    (
        gene2_without_copied_localizations,
        returns2_without_copied_localizations,
    ) = remove_values_from_permutation(
        gene2, returns2, gene1[copy_start_index : copy_end_index + 1]
    )

    new_gene[:copy_start_index] = gene2_without_copied_localizations[:copy_start_index]
    new_gene[copy_end_index + 1 :] = gene2_without_copied_localizations[
        copy_start_index:
    ]

    new_returns[:copy_start_index] = returns2_without_copied_localizations[
        :copy_start_index
    ]
    new_returns[copy_end_index + 1 :] = returns2_without_copied_localizations[
        copy_start_index:
    ]

    return new_gene, new_returns


@jit(nopython=True)
def remove_values_from_permutation(
    array: np.ndarray, array2: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    result1 = np.empty(len(array) - len(values), dtype=np.int32)
    result2 = np.empty(len(array) - len(values), dtype=np.int32)

    current_index = 0
    for x, y in zip(array, array2):
        if x not in values:
            result1[current_index] = x
            result2[current_index] = y
            current_index += 1

    return result1, result2
