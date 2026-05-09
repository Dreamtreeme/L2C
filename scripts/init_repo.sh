#!/usr/bin/env bash
# =====================================================
# L2C - 초기 git 셋업 스크립트 (macOS/Linux/WSL)
# =====================================================
set -e

for d in .git .git_broken; do
  if [ -d "$d" ]; then
    echo "[cleanup] removing $d ..."
    rm -rf "$d"
  fi
done

echo "[init] git init -b main"
git init -b main >/dev/null

git config user.name "Dreamtreeme"
git config user.email "k99702@gmail.com"

git add .
git commit -m "chore: initial project structure and README"

git tag -a v0.0-init -m "Initial project structure"

REMOTE="https://github.com/Dreamtreeme/L2C.git"
if git remote | grep -q origin; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

echo ""
echo "===================================="
echo " 셋업 완료"
echo "===================================="
git log --oneline
git tag
echo ""
echo "다음 단계:"
echo "  1) GitHub에서 빈 레포 생성 (https://github.com/new), name=L2C"
echo "  2) git push -u origin main"
echo "  3) git push origin v0.0-init"
