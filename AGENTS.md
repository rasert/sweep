# Project Charter & Diretrizes Agênticas (Spec-Anchored): Paridade SWeeP

> **Projeto:** Paridade Funcional e Rigor Matemático `rSWeeP` (R) → `sweep` (Python)  
> **Instituição:** PPGBioinfo / UFPR — Laboratório de Inteligência Artificial Aplicada à Bioinformática (AIBIA Lab)  
> **Status:** Spec-Anchored v2.2.0  

---

## 1. Os 6 Invariantes Matemáticos Obrigatórios (Invioláveis)

1. **Base Quase-Ortonormal Determinística (`orth_base` / `sweep_lite`):**
   * Deve utilizar os **50 primeiros números primos** (`2..229`).
   * PRNG exato `R-MT19937`: `set.seed(seed)` executa **51 passos de warm-up** no LCG `69069 * s + 1` antes de preencher `MT19937` e gerar `Mproj` ($50 \times \text{col}$) em ordem *column-major* (`order="F"`).
   * Fórmula modular 1-based: $(j + 1) \bmod P_p$.
2. **Dupla Formulação Preservada:**
   * `orth_base()` e `sweep_lite()` usam os 50 primos do `rSWeeP` (v2.9).
   * `orthbase()` legado (SVD) e `get_default_proj_mat()` (SRB 2020) permanecem funcionais e testados para reprodução histórica.
3. **Indexação Bijetiva e Descarte Estrito de Janelas:**
   * $f(S_k) = 1 + \sum_{q=1}^m (V(S_q) - 1) \cdot K^{q-1}$.
   * Qualquer janela com resíduo fora de $[1, K]$ ou `*` em **posição ativa (`mask == 1`)** é descartada.
   * Caracteres especiais em posições neutras (`mask == 0`, *don't care*) preservam a janela.
   * Multi-FASTA concatenado com `spacer = "*" * len(mask)`.
4. **Tabela de Verdade dos 8 Modos e Zero-Materialização:**
   * `bin=False, norm="logNeg"`: $x=0 \mapsto -1$, $x=1 \mapsto \log_{10}(1) = 0$, $x>1 \mapsto \log_{10}(x)$.
   * Identidade algébrica esparsa: $\tilde{X} B = \hat{X}_{\text{sparse}} B - \mathbf{1}_N (\mathbf{1}_L^T B)$, onde entradas não nulas recebem $(\hat{X}_{\text{sparse}})_{k, i} = v_{k, i} + 1$ (para $x=1$, vale $+1$, cancelando exatamente o $-1$ do offset sem densificar zeros em RAM/VRAM).
5. **Precisão `np.float64` (IEEE 754):**
   * Todas as novas funções científicas operam em `float64` tanto em CPU quanto em GPU.
6. **Topologia Exata em Filogenia (`pcci` / `pmpg`):**
   * `pcci`: `edge.length = 1.0` (distância cofenética topológica), preservação da ordem de folhas e desempate decrescente de `binvec`.
   * `pmpg`: regra de enraizamento exata de `ape::is.monophyletic` (`reroot = !is.rooted(phy)` via outgroup) e poda iterativa `ape::drop.tip`.

---

## 2. Matriz de Arquivos por Trilha / Git Worktree

| Worktree | Branch | Arquivos Permitidos |
| :--- | :--- | :--- |
| **Worktree A** (`sweep-wt-hdv`) | `feat/wt-hdv-indexing` | `sweep/indexing.py`, `sweep/hdv.py`, `tests/unit/test_checks.py`, `tests/unit/test_masks_and_indexing.py`, `tests/unit/test_hdv_and_norm.py`, `tests/integration/test_extract_hdv.py` |
| **Worktree B** (`sweep-wt-proj`) | `feat/wt-projection-gpu` | `sweep/r_mt19937.py`, `sweep/backend.py`, `sweep/projection_core.py`, `tests/unit/test_projection_core.py`, `tests/integration/test_orthbase.py` |
| **Worktree C** (`sweep-wt-phylo`) | `feat/wt-phylo` | `sweep/phylo.py`, `tests/unit/test_phylo_internals.py`, `tests/integration/test_phylo.py` |
| **Worktree D / Onda Final** | `feat/wt-core-sweep` | `sweep/core.py`, `sweep/__init__.py`, `tests/integration/test_sweep_fasta.py`, `tests/integration/test_sweep_matrix.py`, `tests/unit/test_mathematical_invariants.py`, `tests/unit/test_legacy_deprecation.py` |

---

## 3. Portões de Qualidade Obrigatórios

Todo código desenvolvido deve passar com 0 erros:
```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run mutmut run --max-children 8
```
