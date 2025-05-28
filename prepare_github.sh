#!/usr/bin/env bash
set -euo pipefail

# Usage: ./prepare_github.sh [repo-slug] [public|private]
# Example: ./prepare_github.sh spectral-entropy-prime-predictor public

# 1. Parse arguments
REPO_SLUG=${1:-spectral-entropy-prime-predictor}
VISIBILITY=${2:-public}

# 2. Full human-readable title for README
FULL_TITLE="Spectral-Entropy-Guided Domain-Weighted Regression for Microsecond-Scale Prime Discovery"

echo "🔧 Scaffolding project for GitHub as '$REPO_SLUG' ($VISIBILITY)…"

# 3. Create necessary directories
mkdir -p src configs tests .github/workflows

# 4. Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.env
venv/

# Logs
*.log

# macOS files
.DS_Store

# Windows files
Thumbs.db

# IDE / Editor folders
.vscode/
.idea/
EOF

# 5. Create requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.25.0
scipy>=1.11.0
pyyaml>=6.0
matplotlib>=3.7.0
pytest>=7.4.0
EOF

# 6. Create README.md
cat > README.md << EOF
# $FULL_TITLE

_Repository slug:_ \`$REPO_SLUG\`

**Meta-Cognitive Philosopher · Theoretical Physicist · AI-Neuroarchitect**

---

## Abstract
This repository contains the initial implementation of “Spectral-Entropy-Guided Domain-Weighted Regression for Microsecond-Scale Prime Discovery,”  
a hybrid framework combining emergent-entropy learning with quantum-inspired neural layers.

---

## Installation

\`\`\`bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
\`\`\`

---

## Usage

\`\`\`bash
python simulate.py --config configs/default.yaml
\`\`\`

**Example LaTeX expression:**  
$$
\tilde{\mathcal{V}}_{x_n^{(i)}} \;=\;
\frac{\displaystyle \int_{\aleph_0}^{2^{\aleph_0}} \! dx}{X_n}
\;=\;\frac{100\%}{X_n}
$$

---

## Project Structure

$REPO_SLUG/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── simulate.py
├── src/
│ └── your_module.py
├── configs/
│ └── default.yaml
├── tests/
│ └── test_basic.py
└── .github/
└── workflows/
└── ci.yml


- **simulate.py**: Entry point for simulations.  
- **src/**: Code for core modules (e.g., Domain-Weighted Regression).  
- **configs/**: YAML configuration files.  
- **tests/**: Unit tests.  
- **.github/workflows/ci.yml**: Continuous Integration pipeline.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
EOF

# 7. Create LICENSE (MIT)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Budd McCrackn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
EOF

# 8. Create simulate.py
cat > simulate.py << 'EOF'
#!/usr/bin/env python3
import yaml

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    print("Running simulation with config:", cfg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the prime discovery simulation")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
EOF
chmod +x simulate.py

# 9. Create src/your_module.py
cat > src/your_module.py << 'EOF'
def core_function(x):
    """
    TODO: Implement the spectral-entropy-guided regression core.
    """
    return 42
EOF

# 10. Create configs/default.yaml
cat > configs/default.yaml << 'EOF'
param1: 0.1
param2: 100
EOF

# 11. Create tests/test_basic.py
cat > tests/test_basic.py << 'EOF'
import pytest
from src.your_module import core_function

def test_core_function_returns_expected():
    assert core_function(0) == 42
EOF

# 12. Create GitHub Actions CI workflow
cat > .github/workflows/ci.yml << 'EOF'
name: Continuous Integration

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --maxfail=1 --disable-warnings -q
EOF

# 13. Initialize Git and make the first commit
if [ ! -d .git ]; then
  git init
fi
git add .
git commit -m "chore: initial scaffold with docs, tests, and CI workflow"

# 14. Create GitHub repo & push (if gh CLI is available)
if command -v gh &> /dev/null; then
  echo "🔗 Creating GitHub repository '$REPO_SLUG' ($VISIBILITY)…"
  gh repo create "$REPO_SLUG" --"$VISIBILITY" --source=. --remote=origin --push
else
  echo "⚠️  GitHub CLI not found. Please manually create a repo named '$REPO_SLUG', then:"
  echo "    git remote add origin git@github.com:YOUR_USERNAME/$REPO_SLUG.git"
  echo "    git branch -M main"
  echo "    git push -u origin main"
fi

echo "✅ Project is now GitHub-ready!"
