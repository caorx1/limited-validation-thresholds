# Source attribution and changes

The five official UCI source records below were checked on 2026-10-01 and each labels the dataset CC BY 4.0. Attribution follows UCI citation metadata; citation years are not necessarily donation dates. Original data providers do not endorse this study.

| Dataset | UCI citation attribution | DOI / official source |
| --- | --- | --- |
| HTRU2 | Lyon, R. (2015) | https://doi.org/10.24432/C5DK6R / https://archive.ics.uci.edu/dataset/372/htru2 |
| Default of Credit Card Clients | Yeh, I. (2009) | https://doi.org/10.24432/C55S3H / https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Iranian Churn | Iranian Churn (2020); no creator named in retrieved citation | https://doi.org/10.24432/C5JW3Z / https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |
| Online Shoppers Purchasing Intention | Sakar, C. and Kastro, Y. (2018) | https://doi.org/10.24432/C5F88Q / https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Internet Advertisements | Kushmerick, N. (1999) | https://doi.org/10.24432/C5V011 / https://archive.ics.uci.edu/dataset/51/internet+advertisements |

Changes: official source records were processed into ordered model inputs, with feature-pattern grouping and training/validation/test partitions. Default Credit ID, Iranian Churn Customer Value and Online Shoppers PageValues were excluded from primary prediction. This release contains derived test probabilities, binary test labels and final-threshold records, not raw predictors or source identifiers. Derived paired Macro-F1 gains and source-level comparison summaries are supplied. Internal bootstrap thresholds, sampled validation inputs and training implementation are excluded from this minimal release. Original result and frozen-manifest hashes preserve links to the archived study; per-file hashes describe the reduced files, not the original full caches.

License: https://creativecommons.org/licenses/by/4.0/ . The source-page statements are recorded as evidence, not a certification of all possible third-party rights.
