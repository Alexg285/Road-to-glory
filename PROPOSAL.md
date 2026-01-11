# Project Proposal

## Title
Predicting Team Performance in FIFA World Cups Using Machine Learning

## Motivation
Predicting the performance of national teams in major international football tournaments
such as the FIFA World Cup is a challenging task due to limited data availability, high
uncertainty, and strong structural constraints imposed by tournament formats.
Unlike domestic leagues, World Cups are short competitions with few matches per team,
which makes traditional prediction approaches less reliable.

With the increasing availability of historical football data, machine learning techniques
offer new opportunities to model team performance beyond simple win–loss statistics.
However, most existing work focuses on match-level prediction, while fewer studies address
tournament-level progression.

This project aims to explore whether machine learning models can predict how far a team
will progress in a World Cup, rather than predicting individual match outcomes.

## Research Question
Can the stage reached by national teams in FIFA World Cups be predicted using a combination
of group-stage performance indicators and pre-tournament contextual information?

## Data
The project relies on publicly available datasets covering multiple FIFA World Cup editions.
These include:
- Match-level World Cup results
- Team squad and player information
- Historical FIFA rankings
- Historical Elo ratings
- Final tournament rankings (winner, runner-up, etc.)

All datasets will be merged to construct one observation per team per World Cup.

## Methodology
The prediction task is formulated as an ordinal classification problem, where the target
variable represents the stage reached by a team (group stage, Round of 16, quarterfinals,
semifinals, final, winner).

Two types of feature sets will be considered:
- Baseline features based on group-stage performance (points, goal difference, wins, etc.)
- Enriched features including ranking-based and contextual variables (FIFA/Elo ratings,
  host status, average team age)

Two machine learning models will be implemented and compared:
- Multinomial logistic regression as an interpretable baseline
- Gradient boosting (XGBoost) to capture non-linear relationships

Model performance will be evaluated using accuracy and macro-averaged F1-score to account
for strong class imbalance.

## Expected Contributions
This project aims to:
- Provide a reproducible framework for tournament-level performance prediction
- Evaluate the incremental value of contextual and ranking-based features
- Highlight the challenges of ordinal prediction in small, imbalanced sports datasets

## Status
This proposal was submitted earlier in the course.
The final implementation, results, and analysis are presented in the accompanying project
report.