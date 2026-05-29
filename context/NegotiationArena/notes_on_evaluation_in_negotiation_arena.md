
# Evaluation

## Metrics

* **Win Rate:** The percentage of decisive games where one agent secures more resources than the other agent. Games ending in a tie are ignored.
* **Payoff:** The average number of resources each agent holds after the trade is concluded.

## Game Scenarios
### Resource Exchange (Trading Game)

* Each agent has access to a set of resources and a specific goal.
* Both agents use multiple turns to make proposals until one accepts.

* **Ending condition:** Acceptance by either player or reaching the maximum number of turns (8 rounds).
* **Winning condition:** One agent achieves a higher net increase in total resources than their opponent.
* **Payoff:** Final Resources minus Initial Resources.

### Ultimatum Game

* A classical game used to study human behavior, specifically fairness and rationality.
* One agent is given all the game's resources and proposes a split. The second agent either accepts or rejects; if rejected, both agents lose all resources.
* While the classical Ultimatum game only has one round, the *Negotiation Arena* version allows multiple turns for counteroffers.

* **Ending condition:** Acceptance by either player or reaching the maximum number of turns (8 rounds).
* **Winning condition:** The agent ends up with more absolute resources than their opponent.
* **Payoff:** The absolute amount of money an agent walks away with.

### Seller and Buyer (BuySell)

* An incomplete information game involving two agents: one looking to sell a set of resources (Seller) and one looking to buy them (Buyer).
* The seller (Player 1) is aware of the **production cost**.
* The buyer (Player 2) is aware of their **willingness to pay** for the object.

* **Ending condition:** Acceptance by either player or reaching the maximum number of turns (**10 rounds**, not 8).
* **Winning condition:** The agent who achieves a higher payoff (economic surplus).
* **Payoff:** * *Buyer's payoff:* Willingness to pay minus the final agreed-upon price.
* *Seller's payoff:* Final agreed-upon price minus the cost of production.

---

## Section 1: Cross-Play Interactions (Impact on Win Rate and Payoff)

*In this section, different models were pitted against each other to evaluate relative capabilities.* 

### Trading Game (Resource Exchange):

* The **agent going second** tends to **beat the first agent**.
* GPT-4 and Claude-2.1 tend to be the best overall negotiators in this scenario, while GPT-3.5 is the worst.
* While GPT-4 wins more often as Player 2, **Claude-2.1 achieves a higher average payoff** in that role. This is because GPT-4 aggressively tries to "diversify" and will trade abundant resources at a loss just to acquire a scarce one.

### Ultimatum Game:

* **Player 1** almost **always wins**.
* Claude models tend to achieve a higher payoff than GPT models. This occurs because Claude agents consistently propose initial splits that are roughly 10 dollars lower than GPT agents, leaving them more room for negotiation.
* **The "Babysitting" Effect:** GPT-3.5 frequently acts illogically or proposes impossible trades. When powerful models like GPT-4 play against it, they get distracted trying to correct GPT-3.5's errors, which detracts them from their actual goals and lowers their overall payoff.

### Seller and Buyer:

*(Assuming a setup of 40 cost of production / 60 willingness to pay)* 

* In this setup, LLM agents consistently perform better as buyers than as sellers.
* All LLMs achieve very similar final prices when playing as the *Seller*. GPT-4 only stands out as the definitively best model when playing as the *Buyer*. This is likely because the buyer role inherently allows for more flexible negotiation tactics.

---

## Section 2: Effects of Social Behavior on Agents

*In this section, Player 1 was a default GPT-4 agent, while Player 2 was prompted with specific personas (Cunning, Desperate, or Default).* 

### Trading Game (Resource Exchange):

* Adopting social behaviors (both Cunning and Desperate) generally leads to an increase in both the average payoff and the win rate for the agent utilizing them.

### Seller and Buyer:
* **The Buyer (Player 2) always wins** because the final agreed-upon price is consistently lower than the midpoint (50).
* Utilizing personas slightly increases the Buyer's overall payoff.

### Ultimatum Game:

* Personas have a massive impact here. Without any social behavior, Player 2 is almost never able to obtain more than 50 dollars from Player 1.
* Prompting the model to be desperate or cunning drastically increases Player 2's frequency of winning.
* **The Risk of Cunning:** While the "Cunning" persona (insulting the opponent) increases Player 2's win rate to 82%, it is a **high-reward, high-risk tactic**. The cunning player's average overall payoff barely improves compared to the default because their hostile tactics frequently cause negotiations to break down entirely, resulting in a 0 payoff for both players.

---

## Section 3: Irrationality and Cognitive Biases (New)

*LLMs do not negotiate purely rationally; they exhibit distinct cognitive biases that heavily mirror human psychology.* 

* **Anchoring Bias:** The initial price proposed in a negotiation acts as a strong anchor. There is a very strong correlation between the first price proposed and the final agreed-upon price.

* **Blindly "Splitting the Difference":** Both buyer and seller agents rely heavily on proposing a new price that splits the difference of the two most recent proposals. They will do this even when it makes no rational sense—for example, if a Buyer's willingness to pay is artificially inflated, they are highly likely to counter-offer with a *higher* price than the seller initially asked for, simply because the LLM feels it must "meet in the middle".

* **Scale / Denomination Bias:** LLM strategies change based purely on the absolute numbers used, even if the underlying rational strategy shouldn't change. For example, as the total pool of money to split in an Ultimatum game is scaled up (e.g., to 10,000,000,000 dollars), the decider becomes far more likely to accept a highly unfair split simply because the absolute sum of money is so large.