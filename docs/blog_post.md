# From Random to Contextual: Comparing Bandit Algorithms for News Recommendation

## Motivation

Every news website faces the same problem: a user arrives, the system has a handful of candidate articles, and it must decide which one to show. If the user clicks, that's a success. If not, an opportunity was wasted. The goal is straightforward — maximize the click-through rate (CTR) over time — but the path to getting there is not.

The core challenge is the **explore/exploit dilemma**. Should the system recommend the article it currently believes is best (exploit), or should it try something less certain to learn more about what works (explore)? Pure exploitation misses potentially great articles it has never tried. Pure exploration wastes impressions on articles that have already proven ineffective. Every recommendation system must navigate this tradeoff.

Traditional A/B testing is one approach: split users into groups, show each group a different article, and after collecting enough data, pick the winner. But A/B tests are static. They require pre-defined variants, run to completion before any learning happens, and start over from scratch with each new test. In a domain like news — where content changes hourly and user preferences shift — this rigidity is costly.

**Bandit algorithms** offer a better model. They learn in real time, continuously adapting their recommendations based on observed feedback. Instead of waiting for a test to finish, a bandit shifts traffic toward better-performing options while still exploring alternatives. The learning never stops.

Within the bandit framework there is a spectrum of sophistication. **Multi-armed bandits** (MABs) treat every user the same — they learn which articles are globally popular but cannot personalize. **Contextual bandits** go further, using information about the user and the article to make personalized decisions. A sports fan and a finance reader should see different recommendations, and contextual algorithms can learn these preferences.

In this post we compare four algorithms that span this spectrum, from a random baseline to a fully contextual approach. We evaluate them head-to-head on the Microsoft News Dataset (MIND), a collection of real user impression logs. The question: how much does each level of sophistication actually buy you?

## Methods

All four algorithms face the same decision each round: given a set of candidate articles, pick one. After the user either clicks or ignores the recommendation, the algorithm updates its internal state. The difference between algorithms is *how* they make their choice and *what* they learn from the outcome.

### RandomChoice (Baseline)

The simplest possible strategy: pick an article uniformly at random with no regard for past performance, user history, or article content.

**Strengths:** Dead simple and completely unbiased. It makes no assumptions and requires no tuning. Its only purpose is to establish a performance floor — any useful algorithm must beat this.

**Weaknesses:** It ignores everything. Every click it gets is pure luck. With ~6.6 candidates per round, we'd expect a CTR around 15%, which is exactly what it delivers.

### Epsilon-Greedy

Epsilon-greedy introduces the most basic form of learning. It tracks the average reward (click rate) for each article it has tried. With probability 1-ε it *exploits* by picking the article with the highest known average. With probability ε it *explores* by picking randomly. We use ε=0.1, meaning it exploits 90% of the time.

**Strengths:** Easy to understand and implement. It learns quickly which articles are popular across the user base. The ε parameter gives direct, intuitive control over the explore/exploit balance — turn it up for more exploration, down for more exploitation.

**Weaknesses:** Its exploration is blind. When it does explore, it picks randomly among all candidates, including ones that have already proven poor. It also treats all users identically — a sports fan and a foodie see the same recommendations. And it never stops exploring, even when it is highly confident about which articles are best. That 10% random exploration continues forever, wasting impressions on known losers.

### Thompson Sampling

Thompson Sampling takes a Bayesian approach to the explore/exploit problem. Instead of tracking a single average per article, it maintains a full probability distribution (a Beta distribution) over each article's true click rate. At decision time, it draws a random sample from each article's distribution and picks the article whose sample is highest.

**Strengths:** Exploration is principled and automatic. Articles that the algorithm is uncertain about have wide distributions — their samples sometimes come out high, naturally triggering exploration. Articles with strong evidence of high click rates have narrow distributions centered on good values, so they consistently win. There is no exploration parameter to tune. As confidence grows, exploration naturally tapers off.

**Weaknesses:** Like epsilon-greedy, Thompson Sampling is context-free. It cannot personalize. More subtly, it is sensitive to the choice of prior distribution in sparse settings. With 200 articles and only 500 impression rounds, each article gets pulled just 2-3 times on average. A uniform prior — Beta(1,1), expressing total ignorance — means the posterior is barely different from the prior after so few observations. Thompson Sampling effectively degrades to random selection.

**Key insight:** We solved this by using a pessimistic prior of Beta(1,8), which encodes a prior belief that articles are clicked about 11% of the time (close to the actual base rate). With this prior, even a single click on an article meaningfully shifts its posterior upward, allowing the algorithm to exploit that signal. Prior tuning is a practical consideration that textbooks often gloss over, but it makes or breaks Thompson Sampling in real-world sparse settings.

### LinUCB (Contextual Bandit)

LinUCB, introduced by Li et al. at WWW 2010, is a contextual bandit algorithm. Unlike the previous approaches, it does not treat articles or users as interchangeable. Instead, it builds a linear model for each article that predicts click probability from a **context vector** — a numerical description of the user and the article.

At decision time, LinUCB computes a predicted reward for each candidate *plus* an "optimism bonus" that is large when the algorithm is uncertain about a prediction. It picks the article with the highest optimistic estimate. This is the Upper Confidence Bound (UCB) principle: be optimistic in the face of uncertainty, because the upside of discovering a great option outweighs the cost of a bad trial.

The context vector we use is 20-dimensional: 10 dimensions for the **user profile** (a normalized distribution over news categories based on which articles the user has clicked in the past) and 10 dimensions for the **article category** (a one-hot encoding). This lets LinUCB learn patterns like "users who read sports articles tend to click on other sports articles."

**Strengths:** LinUCB personalizes. Different users get different recommendations based on their reading history. It shares learning across contexts — a click on a sports article by a sports-oriented user informs predictions for similar user-article pairs, not just that specific article. And its exploration bonus shrinks automatically as confidence grows, so it explores less over time without requiring manual tuning.

**Weaknesses:** LinUCB is only as good as the features you give it. The algorithm itself is elegant, but feature engineering is the real bottleneck.

**Key insight:** With article-only features (a 10-dimensional category one-hot), LinUCB actually underperformed epsilon-greedy. It was learning "which categories get clicked globally" — something epsilon-greedy already captures by tracking per-article averages. Adding the user profile from click history was the critical change. The algorithm needs a user-article interaction signal to do what it was designed to do: personalize.

## Experiment

### Dataset

We evaluate on MINDsmall_dev, a development split from the Microsoft News Dataset (MIND). It contains 500 user impression rounds across 200 unique news articles spanning 10 categories: news, sports, entertainment, finance, lifestyle, health, autos, travel, food & drink, and weather. Each impression presents approximately 6.6 candidate articles on average, with ground-truth click labels indicating which article the user actually engaged with.

### Simulation

We replay impressions sequentially through a simulation engine. For each round, the algorithm sees the candidate articles (and, for LinUCB, the context vectors), selects one article to recommend, and then observes the ground-truth reward: 1.0 if the user clicked that article, 0.0 otherwise. The algorithm updates its internal state and moves to the next round.

Critically, all four algorithms process the same 500 impressions in the same order. This ensures a fair apples-to-apples comparison — differences in CTR reflect algorithmic merit, not data ordering.

### Context Construction

For LinUCB, we construct a 20-dimensional context vector per candidate article in each round. The first 10 dimensions encode the **user profile**: a normalized category-frequency vector built from the articles the user has previously clicked. A user who clicked 3 sports articles and 1 finance article would have high weight on sports and low weight elsewhere. The remaining 10 dimensions encode the **article category** as a one-hot vector. This concatenated representation gives LinUCB the user-article interaction signal it needs for personalization.

### Metric

Our primary metric is click-through rate (CTR): total clicks divided by total impressions. We also track cumulative CTR at each round to visualize how algorithms learn over time.

## Results

### Final Performance

| Algorithm | CTR | vs. Random |
|---|---|---|
| RandomChoice | 0.1660 | — |
| EpsilonGreedy | 0.1940 | +16.9% |
| ThompsonSampling | 0.1940 | +16.9% |
| LinUCB | 0.1960 | +18.1% |

![Final CTR comparison across all four algorithms](../results/final_ctr.png)

All three learning algorithms meaningfully outperform the random baseline. LinUCB edges ahead as the top performer, though the margin over epsilon-greedy and Thompson Sampling is slim — just one additional click out of 500 impressions.

### Learning Curves

![Cumulative CTR over 500 impression rounds](../results/cumulative_ctr.png)

The cumulative CTR plot reveals how each algorithm learns over time. The early rounds (1-50) are noisy due to small sample sizes. RandomChoice quickly converges to ~16.6% and stays flat — there is no learning happening. The three learning algorithms begin to separate from random after roughly 50-100 rounds as they accumulate enough feedback to start making informed decisions. LinUCB shows a slightly higher CTR through the middle of the run (rounds 100-300), consistent with its ability to leverage user context. By round 500, the learning algorithms cluster tightly between 19.4% and 19.6%.

### Takeaways

**1. Any learning beats no learning.** Even the simplest learning algorithm, epsilon-greedy, achieves a ~17% relative improvement over random. The explore/exploit framework delivers real value with minimal complexity.

**2. Bayesian priors need tuning in sparse settings.** Thompson Sampling is mathematically elegant, but with 200 articles and only 500 rounds, the prior distribution matters enormously. A uniform prior made it indistinguishable from random. A pessimistic prior calibrated to the base click rate unlocked its potential. This is the kind of practical detail that separates textbook understanding from working implementations.

**3. Context helps, but features are the bottleneck.** LinUCB's advantage did not come from the algorithm — it came from engineering the right features. Article-only features gave it nothing beyond what simpler algorithms already captured. Adding user preference profiles from click history was the critical unlock. The lesson generalizes: investing in feature engineering often pays more than switching to a fancier algorithm.

**4. Scale limits differentiation.** With 500 impressions, the learning algorithms are nearly indistinguishable. In production systems processing millions of interactions, the compounding effect of better per-impression decisions widens the gap significantly. This small-scale experiment is useful for understanding mechanics but should not be used to draw conclusions about which algorithm is "best" at scale.

**5. The framework matters more than any single result.** A clean algorithmic interface — select an arm, observe a reward, update — makes it trivial to swap in new approaches. The engineering investment in a modular simulation framework pays dividends long after any individual experiment.

### What's Next

This comparison scratches the surface. Natural extensions include scaling to the full MIND dataset (160K+ impressions), enriching features with title embeddings and subcategory information, adding algorithms like UCB1 and contextual Thompson Sampling, and running statistical significance tests to quantify the reliability of observed differences.
