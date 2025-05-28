#!/usr/bin/env bash
set -euo pipefail

# Usage: ./prepare_github.sh [repo-slug] [public|private]
# Example: ./prepare_github.sh spectral-entropy-prime-predictor public

# 1. Parse args
REPO_SLUG=${1:-spectral-entropy-prime-predictor}
VISIBILITY=${2:-public}

# 2. Full human-readable title (will go in README)
FULL_TITLE="Spectral-Entropy-Guided Domain-Weighted Regression for Microsecond-Scale Prime Discovery"

echo "🔧 Scaffolding project for GitHub as '$REPO_SLUG' ($VISIBILITY)…"

# 3. Directories
mkdir -p src configs tests .github/workflows

# 4. .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.env
venv/

# Logs
*.log

# OS files
.DS_Store
Thumbs.db

# IDE / Editor
.vscode/
.idea/
EOF

# 5. requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.25.0
scipy>=1.11.0
pyyaml>=6.0
matplotlib>=3.7.0
pytest>=7.4.0
EOF

# 6. README.md
cat > README.md << EOF
# $FULL_TITLE

_Repository slug:_ \`$REPO_SLUG\`

**Meta-Kognitiv Filosof · Teoretisk Fysiker · AI-Neuroarkitekt**

---

## 📖 Abstract  
Dette repository indeholder første udgave af “The Nothingness Effect” –  
en ramme, der kombinerer emergent entropi-læring med kvanteinspirerede neurale lag.

---

## ⚙️ Installation

\`\`\`bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
\`\`\`

---

## 🚀 Usage

\`\`\`bash
python simulate.py --config configs/default.yaml
\`\`\`

**Eksempel på LaTeX-udtryk:**  
$$
\tilde{\mathcal{V}}_{x_n^{(i)}} \;=\;
\frac{\displaystyle \int_{\aleph_0}^{2^{\aleph_0}} \! dx}{X_n}
\;=\;\frac{100\%}{X_n}
$$

---

## 🗂️ Struktur

- **simulate.py**: Entrypoint til simulering.  
- **src/**: Kode for DFI-moduler og SuperPositionNeuron.  
- **configs/**: YAML-konfigurationer for parametresæt.  
- **tests/**: Grundlæggende enhedstests.  

---

## 📜 License

MIT License – se [LICENSE](LICENSE) for detaljer.
EOF

# 7. LICENSE (MIT)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Budd McCrackn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
EOF

# 8. simulate.py
cat > simulate.py << 'EOF'
#!/usr/bin/env python3
import yaml

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    print("Running hybrid_prime_predicter simulation with config:", cfg)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
EOF
chmod +x simulate.py

# 9. src/your_module.py
cat > src/your_module.py << 'EOF'
def core_function(x):
    # TODO: implement emergent-entropy core
    return 42
EOF

# 10. configs/default.yaml
cat > configs/default.yaml << 'EOF'
param1: 0.1
param2: 100
EOF

# 11. tests/test_basic.py
cat > tests/test_basic.py << 'EOF'
import pytest
from src.your_module import core_function

def test_core_function_returns_expected():
    assert core_function(0) == 42
EOF

# 12. GitHub Actions CI workflow
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on: [push, pull_request]

jobs:
  build:
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

# 13. Git init & first commit
if [ ! -d .git ]; then
  git init
fi

git add .
git commit -m "chore: scaffold project for GitHub with docs, tests & CI"

# 14. Create GitHub repo & push (requires gh CLI)
if command -v gh &> /dev/null; then
  echo "🔗 Creating GitHub repo '$REPO_SLUG' ($VISIBILITY)…"
  gh repo create "$REPO_SLUG" --"$VISIBILITY" --source=. --remote=origin --push
else
  echo "⚠️  gh CLI not found; please manually create a repo named '$REPO_SLUG' and add remote:"
  echo "    git remote add origin git@github.com:\$(gh whoami)/$REPO_SLUG.git"
  echo "    git push -u origin main"
fi

echo "✅ All done! Your project is now GitHub-ready."
