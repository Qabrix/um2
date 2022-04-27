import numpy as np
from numba import jit


@jit(nopython=True)
def cx(
    gene1: np.ndarray, gene2: np.ndarray, returns1: np.ndarray, returns2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    new_gene = np.empty_like(gene1)
    new_returns = np.empty_like(returns1)
    unfilled_indexes: set[int] = set(range(len(new_gene)))

    gene_value_to_index_a = {gene: i for i, gene in enumerate(gene1)}
    gene_value_to_index_b = {gene: i for i, gene in enumerate(gene2)}

    while len(unfilled_indexes) > 0:
        _cx_fill_with_genes_on_cycle(
            parent1_gene=gene1,
            parent2_gene_value_to_index=gene_value_to_index_b,
            unfilled_indexes=unfilled_indexes,
            new_gene=new_gene,
            new_returns=new_returns,
            returns1=returns1,
        )
        _cx_fill_with_genes_on_cycle(
            parent1_gene=gene2,
            parent2_gene_value_to_index=gene_value_to_index_a,
            unfilled_indexes=unfilled_indexes,
            new_gene=new_gene,
            new_returns=new_returns,
            returns1=returns2,
        )
    return new_gene, new_returns


@jit(nopython=True)
def _cx_fill_with_genes_on_cycle(
    parent1_gene: np.ndarray,
    parent2_gene_value_to_index: dict[int, int],
    unfilled_indexes: set[int],
    new_gene: np.ndarray,
    new_returns: np.ndarray,
    returns1: np.ndarray,
):
    if len(unfilled_indexes) <= 0:
        return
    current_gene_index = next(iter(unfilled_indexes))
    while current_gene_index in unfilled_indexes:
        current_gene_value = parent1_gene[current_gene_index]
        new_gene[current_gene_index] = current_gene_value
        new_returns[current_gene_index] = returns1[current_gene_index]
        unfilled_indexes.remove(current_gene_index)
        current_gene_index = parent2_gene_value_to_index[current_gene_value]
