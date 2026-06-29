"""Demo: G-Eval-lite, score contínuo e validação contra rótulos humanos (~1s).

    python src/demo.py
"""

from __future__ import annotations

from pathlib import Path

from scipy.stats import spearmanr

from geval import (
    JuizDeterministico,
    build_prompt,
    fidelidade_proxy,
    geval_score,
    load_casos,
    nota_ingenua,
)

ROOT = Path(__file__).parent.parent


def main() -> None:
    criterio, casos = load_casos(ROOT / "data" / "avaliacoes.json")
    juiz = JuizDeterministico()

    print("=" * 78)
    print("G-Eval-lite: juiz estruturado, nota contínua, validação humana")
    print("=" * 78)
    print("\nExemplo de prompt do juiz (estrutura do G-Eval):")
    print("-" * 78)
    print(build_prompt(criterio, casos[0].contexto, casos[0].resposta))
    print("-" * 78)

    g, n, h = [], [], []
    print(f"\n{'id':>3}  {'fidel.':>6}  {'g-eval':>6}  {'ingênua':>7}  {'humano':>6}")
    for c in casos:
        d = juiz.distribuicao(c.contexto, c.resposta)
        gv, nv = geval_score(d), nota_ingenua(d)
        g.append(gv); n.append(nv); h.append(c.nota_humana)
        print(f"{c.id:>3}  {fidelidade_proxy(c.contexto, c.resposta):>6.2f}  "
              f"{gv:>6.2f}  {nv:>7d}  {c.nota_humana:>6d}")

    print("\n" + "-" * 78)
    print(f"Spearman (g-eval contínuo) vs humano: {spearmanr(g, h).statistic:.3f}")
    print(f"Spearman (nota inteira)    vs humano: {spearmanr(n, h).statistic:.3f}")
    print("Os dois correlacionam forte. O score contínuo não arredonda, então carrega")
    print("mais informação (a vantagem aparece em bases grandes; Liu et al., 2023).")

    # A armadilha: sobreposição léxica alta NÃO é fidelidade.
    a8 = next(c for c in casos if c.id == "a8")
    d8 = juiz.distribuicao(a8.contexto, a8.resposta)
    print("\n" + "-" * 78)
    print("Armadilha (por que validar contra humano):")
    print(f"  resposta: {a8.resposta}")
    print(f"  proxy léxico dá g-eval {geval_score(d8):.2f}, mas o humano deu {a8.nota_humana}.")
    print("  A resposta reusa palavras do contexto e ainda assim o CONTRADIZ.")
    print("  Proxy léxico não pega contradição semântica; um LLM-juiz pega melhor,")
    print("  mas mesmo ele precisa ser validado contra rótulos humanos.")


if __name__ == "__main__":
    main()
