# The Keynesian Beauty Contest

## The Analogy

In *The General Theory of Employment, Interest and Money* (1936), Keynes compared stock market investing to a newspaper contest popular in his era. Readers were shown 100 photographs of faces and asked to pick the six prettiest. The winner wasn't whoever picked the objectively most beautiful faces—it was whoever picked the faces that most closely matched the *average* preferences of all contestants.

## Levels of Strategic Thinking

This creates a recursive problem with escalating levels of sophistication:

**Level 0 (Naive):** Pick the faces *you* find prettiest.

**Level 1:** Pick the faces you think the *average person* finds prettiest.

**Level 2:** Pick the faces you think others think the average person finds prettiest.

**Level 3:** Pick the faces you think others think others think... and so on, infinitely.

The rational player must not only model other players' preferences but model their models of other players' models.

## Application to Financial Markets

Keynes's point: stock prices are not determined purely by fundamentals (earnings, assets, growth). They're determined by what investors *think other investors think* the stock is worth.

Consider:
- A stock's "fair value" based on fundamentals might be $50
- But if everyone believes everyone else will pay $80, buying at $70 is rational
- The price becomes unmoored from underlying value
- You're not betting on the company—you're betting on other people's bets

This explains why:
- Markets can stay irrational longer than you can stay solvent
- Bubbles form even when many participants know prices are inflated
- Crashes happen suddenly when the collective belief shifts

## The Self-Referential Trap

The beauty contest reveals a deep epistemological problem in markets:

1. **No objective anchor**: Unlike physical systems, market "value" is socially constructed
2. **Reflexivity**: Beliefs about prices affect prices, which affect beliefs
3. **Common knowledge matters**: It's not enough to know something—you must know that others know, and know that others know you know

A bubble doesn't pop when smart money realizes prices are too high. It pops when smart money realizes that *everyone else* realizes prices are too high.

## Game-Theoretic Formalization

The beauty contest was later formalized as the "p-beauty contest game":

- Players choose a number between 0 and 100
- The winner is whoever is closest to p × (average of all guesses)
- Typically p = 2/3

**Iterated reasoning:**
- Level 0: Guess 50 (midpoint)
- Level 1: Guess 33 (2/3 × 50)
- Level 2: Guess 22 (2/3 × 33)
- Level ∞: Guess 0 (the Nash equilibrium)

Experimental results show most people reason 1-2 levels deep. Markets are populated by a *distribution* of reasoning depths, which is itself a tradeable edge.

## Implications

**For investors:**
- Technical analysis works not because charts predict the future, but because enough people believe they do
- Narrative and memes can be as important as fundamentals
- Being "right" too early is indistinguishable from being wrong

**For market structure:**
- Coordination mechanisms (indices, benchmarks) become focal points
- Liquidity concentrates where people expect liquidity to concentrate
- This is why "too big to fail" becomes self-fulfilling

**For AI/ML in finance:**
- Models trained on price data are learning the output of a social game, not fundamental relationships
- Alpha decays because strategies become common knowledge
- The meta-game is modeling how other models model the market

## Connection to Other Concepts

- **Schelling focal points**: Coordination without communication
- **Reflexivity (Soros)**: The feedback loop between perception and reality
- **Efficient Market Hypothesis**: Assumes Level-∞ reasoning by all participants—empirically false
- **Greater Fool Theory**: Explicitly Level-1 reasoning as a strategy
- **Mimetic Theory (Girard)**: Desire itself is imitative, not just beliefs about others' desires
