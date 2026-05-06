# Dataset Layout

Place local image datasets here when running ALLIE from an IDE.

Expected structure:

```text
data/
`-- Real_captured/
    |-- Train/
    |   |-- Low/
    |   `-- Normal/
    `-- Test/
        |-- Low/
        `-- Normal/
```

The `Low` folders are the model inputs. The `Normal` folders are the
ground-truth/reference images.

Datasets are intentionally ignored by Git because image datasets are usually too
large for a source-code repository.
