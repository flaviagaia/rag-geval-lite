"""G-Eval-lite: LLM-as-judge estruturado, com nota contínua e validação humana.

O G-Eval (Liu et al., 2023) avalia uma saída em três passos:
  1) descreve o CRITÉRIO em linguagem natural;
  2) pede ao modelo os PASSOS DE AVALIAÇÃO (chain-of-thought) e a nota (form-filling);
  3) em vez de usar só o inteiro gerado, pondera as notas possíveis pelas
     PROBABILIDADES dos tokens, produzindo um score CONTÍNUO mais alinhado ao humano.

Aqui o mecanismo é implementado de forma offline e determinística:
- `build_prompt` monta o prompt real do juiz (critério + passos + caso);
- `Judge` é um protocolo; em produção, um LLM devolve a distribuição de notas
  (via logprobs). Para rodar sem API, `JuizDeterministico` deriva a distribuição de
  um proxy de fidelidade computável (sobreposição léxica resposta/contexto);
- `geval_score` = soma ponderada (contínuo); `nota_ingenua` = argmax (inteiro).

A tese do repo: o score contínuo correlaciona melhor com a nota humana do que o
inteiro. E todo juiz LLM precisa ser VALIDADO contra rótulos humanos (correlação),
nunca usado como verdade absoluta.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

NOTAS = (1, 2, 3, 4, 5)

PROMPT_TEMPLATE = """Você é um avaliador. Critério:
{criterio}

Passos de avaliação:
1. Leia o CONTEXTO (a fonte) e a RESPOSTA.
2. Verifique se cada afirmação da resposta é sustentada pelo contexto.
3. Penalize afirmações inventadas (não sustentadas) e omissões relevantes.
4. Atribua uma nota de 1 (nada fiel) a 5 (totalmente fiel).

CONTEXTO:
{contexto}

RESPOSTA:
{resposta}

Nota (1-5):"""


def build_prompt(criterio: str, contexto: str, resposta: str) -> str:
    """Monta o prompt do juiz (estrutura real do G-Eval: critério + passos + caso)."""
    return PROMPT_TEMPLATE.format(criterio=criterio, contexto=contexto, resposta=resposta)


def _norm(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", t) if len(w) > 2]


def fidelidade_proxy(contexto: str, resposta: str) -> float:
    """Proxy computável de fidelidade: fração das palavras de conteúdo da resposta
    que aparecem no contexto. Em produção, quem julga é o LLM; aqui é o proxy que
    deixa o experimento offline e determinístico."""
    ctx = set(_norm(contexto))
    palavras = _norm(resposta)
    if not palavras:
        return 0.0
    return sum(w in ctx for w in palavras) / len(palavras)


@dataclass(frozen=True)
class Caso:
    id: str
    contexto: str
    resposta: str
    nota_humana: int


class JuizDeterministico:
    """Stand-in offline do LLM-juiz: devolve uma DISTRIBUIÇÃO sobre as notas 1-5,
    centrada na nota implícita pelo proxy de fidelidade. Em produção, troque por um
    LLM que exponha logprobs das notas."""

    def __init__(self, sigma: float = 0.8) -> None:
        self.sigma = sigma

    def distribuicao(self, contexto: str, resposta: str) -> dict[int, float]:
        f = fidelidade_proxy(contexto, resposta)
        centro = 1 + 4 * f  # mapeia [0,1] -> [1,5]
        pesos = {s: math.exp(-((s - centro) ** 2) / (2 * self.sigma ** 2)) for s in NOTAS}
        z = sum(pesos.values())
        return {s: p / z for s, p in pesos.items()}


def geval_score(dist: dict[int, float]) -> float:
    """Nota contínua: soma das notas ponderada pela probabilidade (o truque do G-Eval)."""
    return sum(s * p for s, p in dist.items())


def nota_ingenua(dist: dict[int, float]) -> int:
    """Linha de base: pega só a nota mais provável (inteiro), descartando a distribuição."""
    return max(dist, key=dist.get)


def load_casos(path: Path) -> tuple[str, list[Caso]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    casos = [Caso(id=i["id"], contexto=i["contexto"], resposta=i["resposta"],
                  nota_humana=i["nota_humana"]) for i in data["itens"]]
    return data["criterio"], casos
