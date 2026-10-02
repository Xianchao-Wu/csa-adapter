#########################################################################
# File Name: ignore.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Sep 23 08:51:07 2026
#########################################################################
#!/bin/bash
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/

# Virtual environments
.venv/
venv/
env/

# Logs / temporary files
*.log
log.*
*.out

# IDE
.vscode/
.idea/

# OS
.DS_Store

# Model/cache
.cache/
cache/
data/
checkpoints/
outputs/
runs/
EOF
