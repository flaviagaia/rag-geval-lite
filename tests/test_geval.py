import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr  # noqa: E402

from geval import (  # noqa: E402
    NOTAS,
    JuizDeterministico,
    build_prompt,
    fidelidade_proxy,
    geval_score,
    load_casos,
    nota_ingenua,
)

CRITERIO, CASOS = load_casos(ROOT / "data" / "avaliacoes.json")
JUIZ = JuizDeterministico()


def test_prompt_tem_criterio_contexto_resposta():
    p = build_prompt(CRITERIO, "ctx aqui", "resp aqui")
    assert CRITERIO in p and "ctx aqui" in p and "resp aqui" in p


def test_fidelidade_limites():
    ctx = "o repasse e de cem reais"
    assert fidelidade_proxy(ctx, ctx) == 1.0          # tudo sustentado
    assert fidelidade_proxy(ctx, "xyz abcde fghij") == 0.0  # nada sustentado


def test_distribuicao_e_probabilidade():
    d = JUIZ.distribuicao(CASOS[0].contexto, CASOS[0].resposta)
    assert set(d) == set(NOTAS)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert all(p >= 0 for p in d.values())


def test_score_continuo_no_intervalo():
    for c in CASOS:
        g = geval_score(JUIZ.distribuicao(c.contexto, c.resposta))
        assert 1.0 <= g <= 5.0


def test_geval_monotono_na_fidelidade():
    # Mais fidelidade -> score maior (o contínuo preserva a ordem do sinal).
    ctx = "repasse cem reais por aluno"
    baixo = geval_score(JUIZ.distribuicao(ctx, "bolsa exterior professor viagem"))
    alto = geval_score(JUIZ.distribuicao(ctx, "repasse cem reais por aluno"))
    assert alto > baixo


def test_nota_ingenua_e_argmax():
    d = JUIZ.distribuicao(CASOS[0].contexto, CASOS[0].resposta)
    assert nota_ingenua(d) == max(d, key=d.get)


def test_juiz_correlaciona_forte_com_humano():
    g = [geval_score(JUIZ.distribuicao(c.contexto, c.resposta)) for c in CASOS]
    h = [c.nota_humana for c in CASOS]
    assert spearmanr(g, h).statistic > 0.8  # forte, mas não perfeito


def test_armadilha_overlap_alto_nao_e_fidelidade():
    # a8: reusa palavras do contexto (proxy alto) mas o humano reprova (contradição).
    a8 = next(c for c in CASOS if c.id == "a8")
    g = geval_score(JUIZ.distribuicao(a8.contexto, a8.resposta))
    assert g > 3.0           # o proxy léxico se deixa enganar
    assert a8.nota_humana <= 2  # o humano não
