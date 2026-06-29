# rag-geval-lite

G-Eval na prática: um juiz LLM **estruturado**, com nota **contínua** ponderada por
probabilidade, e a parte que quase ninguém mostra: **validar o juiz contra rótulos
humanos**.

> **Em uma frase:** o G-Eval avalia uma saída descrevendo o critério, pedindo ao
> modelo passos de raciocínio e uma nota, e então pondera as notas possíveis pelas
> probabilidades dos tokens (nota contínua, não o inteiro cuspido); mas todo
> LLM-juiz precisa ser validado contra humanos, porque ele tem vieses e se deixa
> enganar.

> *G-Eval in practice: a structured LLM-as-judge with a probability-weighted
> continuous score (Liu et al., 2023). Here the mechanism runs offline with a
> deterministic stand-in judge, so the demo is reproducible. The headline is
> validation: the judge correlates strongly with human labels, and a built-in
> pitfall shows where it fails, motivating why you must validate.*

---

## O problema

Métricas clássicas (BLEU, ROUGE, BERTScore) comparam a resposta com um gabarito
textual. Em RAG e em texto jurídico isso quase não serve: existem muitas formas certas
de dizer a mesma coisa, e o que importa é se a resposta é **fiel à fonte recuperada**.
A saída foi usar um LLM como juiz. O G-Eval é a forma estruturada de fazer isso.

## Como funciona (o técnico)

O G-Eval tem três passos:

1. **Critério** em linguagem natural (aqui: fidelidade ao contexto).
2. **Passos de avaliação** (chain-of-thought) + a **nota** (form-filling). Veja
   `build_prompt`.
3. **Nota contínua**: em vez do inteiro gerado, pondera as notas 1..5 pelas
   probabilidades dos tokens. `geval_score = Σ nota · P(nota)`.

```
geval_score(dist) = Σ_s  s · P(s)        # contínuo, sem arredondar
nota_ingenua(dist) = argmax_s P(s)       # o inteiro mais provável (linha de base)
```

Para rodar **offline e determinístico**, o LLM-juiz é substituído por um
`JuizDeterministico` que deriva a distribuição de notas de um proxy de fidelidade
computável (sobreposição léxica resposta/contexto). Em produção, troque o juiz por um
LLM que exponha as probabilidades das notas (logprobs); a estrutura é a mesma.

## Resultado (determinístico, offline)

Conjunto fictício com nota humana de fidelidade (1 a 5).

| Medida                                  | Valor   |
| --------------------------------------- | ------- |
| Spearman (g-eval contínuo) vs humano    | 0.93    |
| Spearman (nota inteira) vs humano       | 0.96    |

Os dois correlacionam **forte** com o humano. Nesta base pequena ficam parecidos: o
ganho do score contínuo (não arredondar, carregar mais informação) aparece em bases
grandes, como mostra o artigo original. O valor aqui é o **método** e a **validação**.

### A armadilha (por que validar)

Um item reusa palavras do contexto e ainda assim o **contradiz** ("escolas
particulares com qualquer número de alunos", quando o contexto exige no mínimo
cinquenta). O proxy léxico dá nota alta (~3.5); o humano dá 1. Sobreposição de palavras
não é fidelidade. Um LLM-juiz pega contradição semântica muito melhor que o proxy, mas
**mesmo ele** precisa ser validado contra rótulos humanos antes de virar métrica.

Rode você mesmo:

```bash
pip install -r requirements.txt
python src/demo.py
python -m pytest -q
```

## Como explicar em 30 segundos

"G-Eval é usar um LLM como juiz, de forma organizada: você descreve o critério, pede ao
modelo o raciocínio e a nota, e em vez de pegar só o número você pondera pelas
probabilidades, o que dá uma nota mais fina. Mas juiz LLM tem viés e se engana, então a
regra de ouro é validar a nota dele contra um conjunto rotulado por humanos."

## G-Eval, RAGAS e o ecossistema

- **RAGAS** (ver `rag-evaluation-ragas`): métricas de RAG (faithfulness, relevancy,
  context precision/recall) que, por baixo, são juízes LLM no espírito do G-Eval,
  empacotados para o pipeline retrieval + geração.
- **DeepEval**: popularizou o G-Eval como métrica plug-and-play (critério em uma frase
  vira juiz), com testes estilo pytest.
- **TruLens / Phoenix-Arize**: observabilidade e feedback em produção.
- **ARES**: juízes treinados + inferência com intervalos de confiança estatísticos
  (casa com o rigor estatístico de `rag-evaluation-ragas`).

## Limitações honestas

- O juiz real do G-Eval é um **LLM**; aqui é um proxy léxico determinístico, para o
  experimento rodar sem API. O proxy não capta contradição semântica (ver a armadilha);
  é justamente o que o LLM faria melhor.
- Vieses conhecidos do LLM-juiz: posição (favorece a 1ª opção em comparações par a
  par), verbosidade (premia respostas longas), self-preference (gosta mais da própria
  família de modelo). Por isso a validação humana não é opcional.
- Base pequena e fictícia; as correlações são ilustrativas, não um benchmark.
- A nota humana aqui é sintética (autoral); num estudo real, use vários anotadores e
  reporte concordância entre eles.

## Referências científicas (crédito aos autores)

- **Liu et al. (2023).** *G-Eval: NLG Evaluation using GPT-4 with Better Human
  Alignment.* EMNLP. A técnica deste repo.
- **Es et al. (2024).** *RAGAS: Automated Evaluation of Retrieval Augmented
  Generation.* EACL.
- **Saad-Falcon et al. (2024).** *ARES: An Automated Evaluation Framework for
  Retrieval-Augmented Generation.* NAACL.
- **Zheng et al. (2023).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
  NeurIPS. Vieses do LLM-juiz.
- Dados fictícios; nenhuma relação com dados reais.

Bibliografia completa do portfólio em `REFERENCIAS.md`.

---

Part of my LinkedIn series on efficient RAG → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
