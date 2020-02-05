import numpy as np
from sage.all import *

def show_debug(*v):
    pass


def compute_basis_isotypical_comp(rho_j, rho, G):
    """
    Compute a symmetry adapted basis for the jth component of rho.
    
    Args:
      rho_j: irreducible representation of G
      rho: representation of G
      G: group
    Returns:
      v: [nj x cj x d] array such that 
        v[i, :] = v^i_1 .... v^i_nj a basis of the ith copy 
        of the irreducible reprsentation of rho_j in rho.
        nj is the degree of rho_j, and c_j is the number of copies of rho_j in rho.
        d is the degree of rho.
    """
    d = len(rho(G.an_element()))
    show_debug(LatexExpr(r"\text{order of } \rho = d = "), d)
    nj = rho_j(G.an_element()).dimensions()[0]
    show_debug(LatexExpr(r"n_j = "), nj)
    # QQ(nj) / len(G) *
    pi_j =  matrix(QQ, sum(rho_j(s**-1)[0, 0] * rho(s) for s in G)) * QQ(nj) / len(G)
    cj = rank(pi_j)
    if cj == 0:
        return None
    
    show_debug(LatexExpr(r"\pi_j = "), pi_j, 
         LatexExpr(r", rank(\pi_j) = c_j = "), cj)

    V = VectorSpace(QQ, pi_j.dimensions()[0])
    show_debug(LatexExpr("V = "), V)
    col_space = V.subspace(pi_j.columns())
    col_basis = col_space.basis()
    show_debug("col basis = ", col_basis)
    
    pj_1mu = [  sum(rho_j(s**-1)[mu, 0] * rho(s) for s in G)
                  for mu in range(nj) ]
    pj_1mu = map(lambda M: matrix(QQ, M) * QQ(nj) / len(G), pj_1mu)

    #v = np.zeros((nj, cj, d))
    v = []
    for mu in range(nj):
        PiV = pj_1mu[mu] * matrix(QQ, col_basis).transpose()
        v.append(PiV.transpose())
    return v
