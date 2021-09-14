This folder contains five scripts.

For regression analyses, we need first get words' properties.
```shell
python calc_word_property.py
```

Use ``data_prepare_inter_rater.py`` and ``data_prepare_test_retest.py`` to generate dataframes for the respective regression models 
(inter-rater consistency and test-retest reliability).

Use ``multilevel_inter_rater.R`` and ``multilevel_test_retest.R`` for multilevel modelling.
