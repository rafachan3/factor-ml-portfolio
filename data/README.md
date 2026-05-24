# Data directory

Input CSVs are **not** tracked in git (see `.gitignore`). Symlink or copy them here:

```bash
SRC="/path/to/your/data/folder"
ln -sf "$SRC/mma_sample_v2.csv"    data/mma_sample_v2.csv
ln -sf "$SRC/factor_char_list.csv" data/factor_char_list.csv
ln -sf "$SRC/mkt_ind.csv"          data/mkt_ind.csv
```

Then run `python scripts/check_data.py` to verify integrity.
