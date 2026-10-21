# Task 2 Benchmark: Anticipatory Pre-Failure Warning (Pure Lead Time)

> [!IMPORTANT]
> **Task 2 Scope**: The positive class ($y=1$) consists **strictly of the documented 2-hour pre-failure window** ($[T_{\text{failure}} - 2\,\text{hours}, T_{\text{failure}}]$). **All observations during the active failure outage are explicitly excluded** from the test evaluation. This measures pure prognostic forecasting before fault onset.

| Dataset | Model Architecture | PR-AUC (Primary) | Recall ($	au=0.5$) | F1-Score ($	au=0.5$) | Precision ($	au=0.5$) | Test Pre-Failure Prevalence | Excluded Active Fault Windows | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetroPT-2** | **GRU** | **0.4335** | **1.0000** | **0.6630** | 0.4959 | 0.60% (120 / 20,067) | 3,647 windows | 🥇 Rank 1 |
| **MetroPT-2** | **LSTM** | **0.4026** | **1.0000** | **0.6593** | 0.4918 | 0.60% (120 / 20,067) | 3,647 windows | 🥈 Rank 2 |
| **MetroPT-2** | **TCN** | **0.3644** | 0.9917 | 0.6467 | 0.4798 | 0.60% (120 / 20,067) | 3,647 windows | 🥉 Rank 3 |
| **MetroPT-3** | **GRU** | **0.0207** | 0.0000 | 0.0000 | 0.0000 | 0.51% (120 / 23,667) | 3,258 windows | 🥇 Rank 1 |
| **MetroPT-3** | **TCN** | **0.0177** | 0.0000 | 0.0000 | 0.0000 | 0.51% (120 / 23,667) | 3,258 windows | 🥈 Rank 2 |
| **MetroPT-3** | **LSTM** | **0.0135** | 0.0000 | 0.0000 | 0.0000 | 0.51% (120 / 23,667) | 3,258 windows | 🥉 Rank 3 |

### Methodological Insights on Task 2:
1. **Physical Pre-Failure Symptoms in MetroPT-2**:
   - In MetroPT-2, all three models achieved near-perfect pre-failure recall (**99.2%–100%**) 2 hours before the oil leak began! 
   - However, because the positive class prevalence is only **0.60%** (120 warning windows vs 19,947 normal windows), false alarms during normal operation cap precision at **~49.6%**, yielding a realistic PR-AUC of **0.4335 (GRU)** and **0.4026 (LSTM)**.
   - This proves the models genuinely detected anticipatory signals, while realistically reflecting the difficulty of rare-event detection without inflated active-failure presence.
2. **Cross-Fault Generalization Barrier in MetroPT-3**:
   - In MetroPT-3, training on Air Leaks produced virtually zero anticipatory transfer to the Oil Leak pre-failure window (PR-AUC ~0.014 - 0.021). The pneumatic models could not forecast oil degradation before symptoms physically manifested.
