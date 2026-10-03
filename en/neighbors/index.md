---
synopsis: "Preferential Bayesian optimization shares its likelihood with the reward models that align large language models, and its questions with reward learning, recommender systems, learning to rank, and automated science. This part maps what moves between these fields and what does not."
---

# Neighbors in Computing {#sec-part-neighbors}

Preferential Bayesian optimization shares its likelihood with the reward models used to align large language models, and its questions with reward learning, recommender systems, learning to rank, decision analysis, and automated science. This part maps the traffic between these fields: where one solved a problem first, what moves across, and where an analogy that looks exact breaks down.

The neighbors matter to this book because several of them reached the measurement problem first (@sec-syn-bottleneck). Alignment research measured how far a language-model judge departs from a person, learning to rank learned to correct for the position in which an option is shown, and AI safety formalized systems that change the people they learn from. The first chapter treats large language models in both directions, as components inside a preference loop and as the target of preference learning at a scale no design study reaches. The second visits the other neighbors and collects the problems they solved first that preferential optimization can reuse.

*The part assumes @sec-comparisons and @sec-gp-preference.*
