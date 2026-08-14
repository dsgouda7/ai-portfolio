# External Vehicle Corpora

Raw corpora belong under the ignored `data/external/` directory and must never be committed.

## MIO-TCD

License: CC BY-NC-SA 4.0, non-commercial use. Official source:
https://tcd.miovision.com/challenge/dataset.html

Download the classification and localization archives into `data/external/mio-tcd/`, extract them,
then create an annotation CSV with these columns:

```text
image_path,body_type,camera_id,item_id,geography
```

`camera_id` is mandatory so images from one camera cannot cross train/validation/test boundaries.

## CompCars

License: non-commercial research only; redistribution is prohibited. Official source:
https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html

Follow the official 23-part archive instructions. Create an annotation CSV with:

```text
image_path,body_type,make,model_family,identity_id,item_id
```

`identity_id` is mandatory so views of one physical vehicle cannot cross split boundaries.

## Build a CarFace source manifest

```powershell
python scripts/build_corpus_manifest.py `
  --dataset mio-tcd `
  --dataset-root data/external/mio-tcd/extracted `
  --annotations data-manifests/mio-tcd.csv `
  --output data/vehicles/mio-tcd-manifest.json `
  --accept-noncommercial-terms
```

Use `--dataset compcars` with the corresponding paths for CompCars. The command verifies required
labels, path containment, unique IDs, image existence, SHA-256 provenance, and split ownership.