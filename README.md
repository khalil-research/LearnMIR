To run:

Download the data from the UofT dataverse and put the `goodfiles` directory in the same directory as the scripts.

First, the full separator `01_fullmir.py`. To run the i-th random variation of the j-th instance currently in the `goodinstances` directory:

```
python3 -problem  j -index i
```

This will populate `./goodfiles/j/{cuts,sols,lambdas,base_logs,results}/i/`

Then, the reduced separator `03_reducedmir.py`. To run the i-th random variation of the j-th instance currently in the `goodinstances` directory:

```
python3 03_reducedmir.py -problem  j -index i
```

This will populate `./goodfiles/j/reduced_{cuts,sols,lambdas,base_logs,results}/i/`
