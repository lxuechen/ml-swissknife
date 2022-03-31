### Checks for waterbirds

- `contrastive` beam search
    ```bash
    python -m explainx.waterbird_check \
      --task consensus \
      --beam_search_mode 'contrastive' \
      --contrastive_mode 'marginalization'
    ```

- `mixture` beam search
    ```bash
    python -m explainx.waterbird_check \
      --task consensus \
      --beam_search_mode 'mixture' \
      --contrastive_mode 'marginalization'
    ```

### Checks for CelebA

- `contrastive` beam search
  ```bash
  python -m explainx.celeba_check \
    --task consensus \
    --beam_search_mode 'contrastive' \
    --contrastive_mode 'marginalization' \
  ```
