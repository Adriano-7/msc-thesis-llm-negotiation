### Contrast With NegotiationArena

- The main thesis contrast is that Bianchi et al.'s persona result does not generalize cleanly across models: in this notebook, persona effects are large but highly model-dependent.
- Bianchi et al. report that cunning and desperate behaviors increase both win rate and payoff across their social-behavior games, whereas these OSS self-play runs show gains, losses, and payoff reversals depending on game and model.
- The heatmap is the central evidence for the narrative.
- These runs should not be read as only testing whether P2 can adhere to a persona; they also test how the same model in the default P1 seat reacts to persona wording such as begging, manipulation, and insults.

![](../figures/2_persona_analysis/winrate_delta_heatmap.png)

![](../figures/2_persona_analysis/winrate_by_persona_game.png)

![](../figures/2_persona_analysis/payoff_by_persona_game.png)

### Ultimatum: Win Rate Versus Value Destruction

- Cunning produces the largest conditional P2 win-rate gain in the notebook, raising Ultimatum P2 wins from 0.143 under default to 0.705.
- Cunning does not improve Ultimatum welfare for P2: average P2 payoff falls from 26.19 under default to 16.48 because most games end with no deal.
- The central Ultimatum pattern is a win-rate/payoff split: cunning wins many decided games but destroys enough pots that it becomes the worst payoff condition.
- Desperate is the cleaner positive Ultimatum persona: it raises P2 win rate from 0.143 to 0.230 and payoff from 26.19 to 28.54 without increasing no-deals.

![](../figures/2_persona_analysis/nodeal_by_persona_game.png)

### Ultimatum: Behavioral Mechanism

- Cunning changes the first responder move immediately: P2 accepts only 0.082 of first offers, rejects 0.577, and counter-proposes in 0.341 of games.
- Cunning first counter-proposals are extreme: their median P2 demand is 90, and 0.670 of them demand at least 70 out of 100.
- Across all Ultimatum offers, cunning pushes P2's median demand to 80 while P1's median offer to P2 remains 40, leaving a bargaining gap that the game rarely closes.
- Desperate narrows the Ultimatum gap from both sides: P2's median demand rises to 60, while P1's median offer rises to 40.
- Cunning no-deals are driven mostly by the persona-bearing responder walking away: 165 of 264 completed cunning games end after P2 rejection, compared with 26 after P1 rejection.
- Ultimatum breakdown under cunning is not isolated to one model family: no-deal rates are 0.787 for Gemma, 0.767 for Mistral, and 0.624 for Qwen.

![](../figures/2_persona_analysis/ultimatum_first_p2_action.png)

![](../figures/2_persona_analysis/ultimatum_offer_extremity.png)

![](../figures/2_persona_analysis/ultimatum_rejection_initiator.png)

![](../figures/2_persona_analysis/ultimatum_nodeal_family.png)

### BuySell: Better Terms, Worse Reliability

- Desperate is the strongest BuySell persona overall: P2 win rate rises from 0.752 to 0.967 and average buyer surplus rises from 14.03 to 15.06.
- BuySell desperate improves accepted prices but at a reliability cost: mean accepted price falls from 45.812 to 38.629, while deal rate falls from 0.989 to 0.705.
- BuySell cunning negotiates lower accepted prices than default but destroys too many deals to benefit P2 on average.
- BuySell cunning has a high decided-game win rate of 0.851 but a lower mean payoff than default, because its no-deal rate reaches 0.402.

![](../figures/2_persona_analysis/buysell_deal_price_decomposition.png)

### Trading: No Clear Persona Gain

- Trading does not reproduce the expected persona advantage: desperate lowers P2 win rate from 0.766 to 0.641, while cunning remains close to default at 0.719.
- Trading payoff differences are small relative to the uncertainty, so the notebook does not support a strong persona-effect claim for this game.
- Trading no-deals rise under personas, especially cunning at 0.133 versus 0.020 under default, but the effect is much smaller than the Ultimatum collapse.

### Model Heterogeneity

- Persona effects vary sharply by model rather than shifting all models in the same direction.
- Model heterogeneity is not a secondary detail; it is the main reason the Bianchi et al. persona conclusion needs qualification in this replication setting.
- Ultimatum cunning has the largest average per-model win-rate gain, but also the widest spread: mean +0.297 with spread 1.375 across models.
- Several models convert cunning into large Ultimatum gains, including gemma-3-27b at +0.90, Qwen3.5-27B at +0.80, Qwen3.5-9B at +0.71, and gemma-3-12b at +0.62.
- Cunning is not universally helpful in Ultimatum: gemma-3-4b is roughly flat at -0.06, and Ministral-3-8B-2512 falls by -0.47.
- BuySell cunning is especially model-dependent: its mean effect is only +0.046, but its spread is 1.332, including a -0.681 collapse for Qwen3-14B.
- Persona gains are larger when default P2 is weak, but baseline weakness is only a partial explanation; the pooled baseline-delta correlation is -0.511.

![](../figures/2_persona_analysis/effect_spread_by_model.png)

![](../figures/2_persona_analysis/baseline_vs_persona_delta.png)

### Inconclusive / Weak Signal

- Tier effects should be treated cautiously: Ultimatum cunning is largest at the medium tier, but per-tier decided-game counts are small and the pattern is not monotonic.
- Trading shows no consistent tier ordering for either persona, so model size does not provide a clear narrative there.
- Qwen appears less self-destructive under Ultimatum cunning than Gemma or Mistral, but this is only a descriptive family contrast with three families.
- Baseline weakness helps explain some large gains, especially in Ultimatum, but it cannot explain the mixed BuySell cunning results.

![](../figures/2_persona_analysis/winrate_by_persona_tier.png)
