# 07 - Optimization: The Descent of the Mountain

Now that we know the **direction** to move (the gradient), we need an algorithm to actually move the weights. This is **Optimization**.

## The Mountain Analogy

Imagine you are on top of a mountain (High Loss) and there is a thick fog. You want to reach the valley (Low Loss).
1. Since you can't see the valley, you feel the ground with your feet to find the steepest slope downward.
2. You take a step in that direction.
3. You repeat this until you are at the bottom.

## 1. Gradient Descent (The Basic Step)

This is the simplest version. You update the weight $w$ by subtracting the gradient scaled by a **Learning Rate ($\alpha$)**.

$$ w = w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w} $$

- **Learning Rate ($\alpha$):** 
    - Too large? You might jump over the valley and end up on another mountain.
    - Too small? It will take forever to reach the bottom.

## 2. RMSprop (The Intelligent Step)

Standard Gradient Descent has a problem: it treats all directions the same. If one weight has a huge gradient and another a tiny one, the model might "vibrate" wildly.

**RMSprop** solves this by keeping a "memory" (moving average) of recent gradients. It effectively says:
*"If a gradient has been oscillating wildly, let's slow down. If it's been small, let's speed up."*

$$ v_{dw} = \beta \cdot v_{dw} + (1 - \beta) \cdot (dw)^2 $$
$$ w = w - \frac{\alpha}{\sqrt{v_{dw} + \epsilon}} \cdot dw $$

- **$\beta$:** How much we remember from the past (usually 0.9).
- **$\epsilon$:** A tiny number (like $10^{-8}$) just to make sure we never divide by zero.

## Final Summary: The Learning Loop

1. **Forward Pass**: Give inputs $\rightarrow$ Get prediction.
2. **Calculate Loss**: How far are we from the truth?
3. **Backpropagation**: Calculate the "blame" (gradients) for every weight.
4. **Optimization**: Update weights to reduce loss.
5. **Repeat**: Do this thousands of times until the model is "smart".

---

Congratulations! You have completed the theoretical journey. You are now ready to see how this all looks in **[NumPy code](../neural_network_numpy/model.py)**!
