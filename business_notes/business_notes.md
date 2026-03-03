# 📊 Business Notes – Projeto de Previsão de Churn

## 1. 🎯 Objetivo do Projeto

Desenvolver um sistema preditivo de **Churn de Clientes** capaz de:

- Estimar a probabilidade de cancelamento
- Priorizar clientes com maior risco
- Apoiar decisões estratégicas de retenção
- Permitir simulação individual via aplicação interativa

O modelo foi construído com foco em **recall mínimo controlado**, garantindo alta capacidade de identificação de clientes que realmente irão cancelar.

---

## 2. 📌 Problema de Negócio

A empresa enfrenta:

- Perda recorrente de clientes
- Custo elevado de aquisição de novos clientes
- Falta de priorização nas campanhas de retenção

**Impactos diretos:**

- Redução de receita recorrente
- Aumento do CAC (Custo de Aquisição)
- Ineficiência nas ações de retenção

---

## 3. 💡 Solução Proposta

Implementação de um pipeline completo de Machine Learning com:

- Tratamento automatizado de variáveis categóricas
- Ajuste automático de balanceamento de classes
- Otimização de threshold baseada em recall mínimo
- Avaliação robusta (AUC, KS, Lift por decis)
- Aplicação em produção via interface interativa

---

## 4. 🧠 Estratégia Técnica

### Modelo
- XGBoost com suporte a variáveis categóricas
- Hiperparâmetros previamente otimizados
- Balanceamento automático da classe minoritária

### Estratégia de Threshold
Em vez de utilizar 0.5 como padrão:

- Threshold ajustado para garantir recall mínimo (ex: ≥ 80%)
- Prioriza captura de churners reais
- Trade-off controlado com precision

---

## 5. 📈 Métricas-Chave para Negócio

| Métrica | Interpretação |
|----------|--------------|
| AUC | Capacidade geral de separação |
| KS | Separação máxima entre churn e não churn |
| Recall | % de churners corretamente identificados |
| Precision | Eficiência das ações de retenção |
| Lift (Decis) | Capacidade de priorização comercial |

---

## 6. 💰 Aplicação Estratégica

### Segmentação para Retenção

- Top 10% maior risco → ação imediata
- Top 20% → campanha direcionada
- Baixo risco → monitoramento

### Simulador Individual

Permite:

- Inserir dados de um cliente
- Visualizar probabilidade de churn
- Interpretar impacto das variáveis
- Apoiar decisão comercial com base quantitativa

---

## 7. 🏗 Arquitetura do Projeto

### Treinamento
- Pipeline customizado
- Split Train / Val / Test
- Otimização de threshold
- Salvamento do modelo treinado

### Produção
- Carregamento do pipeline treinado
- Aplicação a novos dados
- Interface interativa para simulação

---

## 8. 📊 Potencial de Impacto Financeiro (Exemplo Ilustrativo)

- Base: 10.000 clientes
- Taxa de churn: 20%
- Ticket médio mensal: R$ 200

Se o modelo recuperar 30% dos churners:

- 600 clientes retidos
- Receita preservada: R$ 120.000/mês
- R$ 1.440.000/ano

*(Valores hipotéticos para demonstração de impacto.)*

---

## 9. ⚠️ Riscos e Cuidados

- Data drift ao longo do tempo
- Mudança de comportamento dos clientes
- Campanhas mal direcionadas podem reduzir margem

**Recomendações:**

- Monitoramento contínuo de métricas
- Re-treinamento periódico
- Avaliação do ROI das ações de retenção

---

## 10. 🚀 Próximos Passos

- Deploy em ambiente cloud
- Implementação de monitoramento de performance
- Integração com CRM
- Testes A/B de campanhas de retenção

---

# 📌 Conclusão Executiva

O projeto entrega:

- ✔ Identificação antecipada de risco
- ✔ Priorização estratégica de retenção
- ✔ Base quantitativa para decisões comerciais
- ✔ Potencial de impacto direto na receita

Trata-se de uma solução escalável que conecta Machine Learning diretamente ao resultado financeiro da empresa.