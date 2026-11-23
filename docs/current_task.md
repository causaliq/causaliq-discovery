# Investigate slow bnlearn performance

I am running workflow @causaliq-research/papers/variable-ordering/workflows/var-ordering.yml
on causaliq-research, which in turn runs bnsl.yml and bnsl-norandomise.yml which in turn
should run the hc algorithm of bnlearn.

I am finding the execution time is growing wildly with sample size - so
that it may be over an hour wth sample sizes over 100,000 rows. Running what in theory
is the same algorithm through the legacy code base, it completes in under 1 second
for all sample sizes upto and including 1,000,000.

Can you investigate what is happening?
