#!/bin/bash
# Pre-push hook: Strip secrets from notebooks before push
# This hook scans all notebooks in commits being pushed and removes API keys/secrets

echo "🔍 Scanning notebooks for secrets before push..."

# Read the pre-push parameters from stdin
# Format: <local ref> <local sha1> <remote ref> <remote sha1>
while read local_ref local_sha remote_ref remote_sha
do
    # If pushing to a new branch, remote_sha will be all zeros
    if [ "$remote_sha" = "0000000000000000000000000000000000000000" ]; then
        # New branch - check all commits
        range="$local_sha"
    else
        # Existing branch - check commits between remote and local
        range="$remote_sha..$local_sha"
    fi

    # Get list of all .ipynb files in the commits being pushed
    notebooks=$(git diff --name-only --diff-filter=ACM "$range" | grep '\.ipynb$' || true)

    if [ -z "$notebooks" ]; then
        echo "✅ No notebooks found in push - skipping secret scan"
        continue
    fi

    echo "📝 Found $(echo "$notebooks" | wc -l) notebook(s) to scan"

    # Track if we found any secrets
    found_secrets=false
    cleaned_files=()

    # Scan each notebook
    for notebook in $notebooks; do
        # Check if file exists (might have been deleted)
        if [ ! -f "$notebook" ]; then
            continue
        fi

        # Check for various secret patterns
        # Pattern 1: API_KEY = "value" or API_KEY="value"
        # Pattern 2: SECRET = "value" or SECRET="value"
        # Pattern 3: AZURE_* = "value"
        # Pattern 4: OPENAI_* = "value"
        # Pattern 5: AWS_* = "value"
        # Pattern 6: WANDB_* = "value"
        
        if grep -qE '(API_KEY|SECRET|AZURE_[A-Z_]*|OPENAI_[A-Z_]*|AWS_[A-Z_]*|WANDB_[A-Z_]*)\s*=\s*["\'][^"\']{8,}["\']' "$notebook"; then
            echo "❌ Found secrets in $notebook"
            found_secrets=true
            
            # Create backup
            cp "$notebook" "$notebook.backup"
            
            echo "   🧹 Stripping secrets..."
            
            # Replace secrets with empty strings
            # Handle various formats: KEY = "value", KEY="value", KEY = 'value', KEY='value'
            sed -i.bak -E 's/(API_KEY\s*=\s*["\'])([^"\']+)(["\'])/\1\3/g' "$notebook"
            sed -i.bak -E 's/(SECRET\s*=\s*["\'])([^"\']+)(["\'])/\1\3/g' "$notebook"
            sed -i.bak -E 's/(AZURE_[A-Z_]*\s*=\s*["\'])([^"\']+)(["\'])/\1\3/g' "$notebook"
            sed -i.bak -E 's/(OPENAI_[A-Z_]*\s*=\s*["\'])([^"\']+)(["\'])/\1\3/g' "$notebook"
            sed -i.bak -E 's/(AWS_[A-Z_]*\s*=\s*["\'])([^"\']+)(["\'])/\1\3/g' "$notebook"
            sed -i.bak -E 's/(WANDB_[A-Z_]*\s*=\s*["\'])([^"\']+)(["\'])/\1\3/g' "$notebook"
            
            # Clean up sed backup files
            rm -f "$notebook.bak"
            
            # Stage the cleaned file
            git add "$notebook"
            cleaned_files+=("$notebook")
            
            echo "   ✅ Cleaned and staged $notebook"
        fi
    done

    if [ "$found_secrets" = true ]; then
        echo ""
        echo "🔐 Secret Removal Summary:"
        echo "   - Cleaned ${#cleaned_files[@]} file(s)"
        echo "   - Files have been automatically staged"
        echo ""
        echo "⚠️  IMPORTANT: The push has been aborted to include the cleaned files."
        echo "   Please run 'git push' again to push the cleaned version."
        echo ""
        
        # Exit with error to abort the push
        # User needs to push again with cleaned files
        exit 1
    fi
done

echo "✅ Secret scan complete - no secrets found"
exit 0
