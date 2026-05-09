# =====================================================
# L2C - 초기 git 셋업 스크립트 (Windows PowerShell)
#
# 사용법:
#   cd C:\Users\psg\Desktop\L2C
#   powershell -ExecutionPolicy Bypass -File scripts\init_repo.ps1
# =====================================================

$ErrorActionPreference = "Stop"

# 0) Linux 샌드박스에서 만든 깨진 .git / .git_broken 폴더가 있으면 삭제
foreach ($d in @(".git", ".git_broken")) {
    if (Test-Path $d) {
        Write-Host "[cleanup] removing $d ..."
        Remove-Item -Recurse -Force $d
    }
}

# 1) git init
Write-Host "[init] git init -b main"
git init -b main | Out-Null

# 2) 사용자 정보 (전역 설정 없으면 기본값)
$existingName = git config user.name 2>$null
if (-not $existingName) {
    git config user.name "Dreamtreeme"
}
$existingEmail = git config user.email 2>$null
if (-not $existingEmail) {
    git config user.email "k99702@gmail.com"
}

# 3) 첫 커밋
Write-Host "[commit] adding all files"
git add .
git commit -m "chore: initial project structure and README"

# 4) 첫 태그
Write-Host "[tag] v0.0-init"
git tag -a v0.0-init -m "Initial project structure"

# 5) 원격 저장소
$remoteUrl = "https://github.com/Dreamtreeme/L2C.git"
$existingRemote = git remote 2>$null
if (-not $existingRemote) {
    Write-Host "[remote] add origin $remoteUrl"
    git remote add origin $remoteUrl
} else {
    Write-Host "[remote] set origin -> $remoteUrl"
    git remote set-url origin $remoteUrl
}

Write-Host ""
Write-Host "===================================="
Write-Host " 셋업 완료"
Write-Host "===================================="
git log --oneline
git tag
Write-Host ""
Write-Host "다음 단계:"
Write-Host "  1) GitHub에서 빈 레포 생성: https://github.com/new"
Write-Host "     - name: L2C"
Write-Host "     - README/.gitignore/license 체크 안함"
Write-Host "  2) git push -u origin main"
Write-Host "  3) git push origin v0.0-init"
