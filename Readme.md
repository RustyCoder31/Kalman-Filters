# Kalman Filters: From Scratch

A from-scratch implementation of the Kalman filter, built in two stages: first a toy "hello world" version on synthetic data to learn the predict/update mechanics, then a real application, using it to track a dynamic hedge ratio between two correlated stocks (Visa and Mastercard). No `pykalman` or other library shortcut; every line of the filter math below is derived and implemented by hand, so I actually understand what's happening at each step instead of just calling a function.

## How to run

Requires Python 3. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

1. `python kalman.py`: runs the hello-world toy filter on synthetic data.
2. `python beta.py`: calibrates, fitting OLS over the 2025-01-01 to 2025-03-03 window and writing `beta`, `R`, `Q` to `model_params.json`. Must be run before step 3.
3. `python pairs_data.py`: loads `model_params.json` and runs the Kalman filter over the out-of-sample period (2025-03-04 to 2026-09-01), producing the beta and price-tracking charts.

## Hello world: toy Kalman filter

For the hello world phase, I have created fake data and included into it noise.
Then I created the for loop for kalman filter with initial Q and R

The filter runs a loop: take the current estimate and its uncertainty, predict forward, then compute the Kalman gain (KG = P / (P + R), where P is the estimate's uncertainty and R is the measurement noise). The gain decides how much the new noisy measurement gets to move the estimate.

KG is large when P is large relative to R: either the filter isn't confident in its own prediction yet, or the measurement is comparatively trustworthy (low R). In that case the filter leans hard on the new reading.
KG is small when P is small relative to R: the filter is already confident in its estimate, or the measurement is noisy (high R), so it mostly ignores the new reading and sticks with what it already believes.

As the filter runs, P shrinks step by step, so KG shrinks and stabilizes too. The filter gets more confident over time and lets each new noisy reading move the estimate less and less, which is why the filtered estimate smooths out the noise instead of following it.

## Kalman Filter on the Pairs (Visa / Mastercard)

### Problem statement

Visa and Mastercard are two correlated stocks: they move together, but there's always some difference between the Visa price and the Mastercard price. How can we predict Visa's price from Mastercard's price? What's the actual relationship between them?

### My theory

We have Visa price `V` and Mastercard price `H`. We relate them as `V = b*H`, and we're trying to guess `b`.

Say our starting guess is `b = 1.5`, and the variance of this guess is `P = 0.1` (this is a squared quantity, and it tells us how uncertain we are about `b`).

We find the residual: `residual = V - b*H`, which tells us how far off our current guess is.

Now we try to figure out who's the "culprit" for this residual: is it because our beta guess is actually wrong, or is it just noise in the price? That's what the Kalman Gain answers:

```
KG = (P*H) / (P*H^2 + R)
```

Here's the units check I worked through: `P*H` gives the predicted uncertainty in terms of price (dollars). `P` is a squared, unitless quantity (variance of a ratio) and `H` is in dollars, so `P*H` is dollars. `R` is a squared quantity (dollars²), so for the denominator to add up correctly, `P*H^2` has to be dollars² too, which it is, since `P` (unitless) times `H^2` (dollars²) gives dollars². So the gain is really comparing "how big is my price-scale uncertainty" against "how noisy is my price measurement."

If KG is high, we're more unsure about our own uncertainty relative to the noise, so we put more weight on the residual and let it move our beta guess.
If KG is low, we assume the residual is mostly measurement noise, so it contributes less to updating beta.

So we update beta and P and loop it: `beta = beta + KG*residual`, `P = (1 - KG*H)*P`.

Beta is the hidden value we're trying to track. At each step we have `beta(t-1)` and its uncertainty `P`. From that we predict tomorrow's `beta(t)`, but we assume `beta(t-1) = beta(t)`, i.e. that it hasn't changed. In reality `beta(t)` might actually be different, so `P_old` can't just carry over unchanged as `P_new`. Instead we inflate it by adding `Q`, precisely because beta is dynamic and its true value might have shifted since yesterday.

- `R` is the variance of the residuals from the regression: how much Visa's actual price deviates from what a fixed beta would predict, i.e. how noisy the price relationship is day-to-day.
- `Q` is the day-to-day variance between the rolling betas: it tells the filter how much to expect beta itself to drift, which is what lets `P` (and therefore `KG`) update correctly each day.

### Process

1. Before I wrote the calibration code, I hand-picked (predicted) values for beta, Q, and R just to see the filter work end-to-end, no regression yet, just plausible numbers plugged into the formulas. That first version actually had a bug: I hadn't added Q into the loop, so P collapsed to ~0 after one big correction and beta froze permanently. Diagnosing that, realizing P has to grow each step via Q precisely because beta is dynamic, is what led me to build the real calibration pipeline in step 2.
2. Then I set up the pipeline to find beta, Q and R properly. Since `visa = beta*mastercard`, beta is just the slope of the line relating both quantities, so I use OLS regression to find beta, and from the predicted Visa prices I find residuals against actual prices and compute their variance, that's R. Then, since Q depends on the day-to-day uncertainty change in beta, I use a 10-day rolling OLS window over the same period (39 trading days), which gives me 30 valid beta values. I find their differences, and the variance of those gives me Q.
3. I plug these values of beta, Q, and R into the Kalman filter to get the results.

### Results

**1. The dynamic beta (`Betas.png`)**

A naive OLS regression over the whole period gives one fixed beta: a flat line. This chart shows why that's a poor model of reality: beta drifts. Starting near 0.63 in March 2025, it drops to about 0.59 by around September/October 2025, plateaus close to 0.61 for the next several months, then climbs sharply to a peak around 0.66 by mid-2026 before easing back down toward 0.645. Because `Q` was calibrated from the actual rolling-beta variance rather than guessed, the filter doesn't overreact to single-day noise (`Q` is tiny relative to `R`, 2.9e-7 vs ~19.7), but it still tracks these slower, multi-month shifts instead of getting stuck.

**2. Price tracking (`Visa vs Mastercard.png`)**

The orange dashed line is the Kalman filter's prediction (`V = beta * H`), the blue line is Visa's actual price. Because beta updates over time instead of staying fixed, the prediction tracks Visa's actual price closely across the full ~1.5-year window, without the permanent drift a fixed-beta model would show once the true relationship shifts.

There are still visible gaps where the actual price pulls away from the prediction for a stretch (e.g. around 2026-05) before the two lines converge again; that's the residual the filter is reacting to. I haven't tested yet whether these gaps are statistically mean-reverting; that's exactly what Phase 2 (cointegration/stationarity testing) is for, next. Right now this shows the filter can track a shifting hedge ratio; it isn't yet evidence of a tradeable arbitrage signal, and no trading logic is built on top of it yet.
