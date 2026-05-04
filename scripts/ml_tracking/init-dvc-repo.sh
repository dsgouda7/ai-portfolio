#!/bin/bash
# Initialize DVC for data version control

echo "📦 Initializing DVC..."

# Initialize DVC
dvc init

# Add DVC files to git
git add .dvc .dvcignore

echo "✅ DVC initialized!"
echo "   Next steps:"
echo "   1. Add remote storage: dvc remote add -d myremote s3://my-bucket/dvc-store"
echo "   2. Track data: dvc add data/raw_data.csv"
echo "   3. Commit: git commit -m 'Initialize DVC'"
