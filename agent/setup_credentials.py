import argparse
import getpass
import sys
import os

from agent.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="L2C Agent 자격 증명(.env) 등록 스크립트")
    parser.add_argument("site", help="등록할 사이트 이름 (예: wanted, jobkorea, saramin)")
    
    args = parser.parse_args()
    site_name = args.site.upper()
    
    print(f"====================================")
    print(f" 🔑 [{site_name}] 로그인 정보 등록 (.env)")
    print(f"====================================")
    
    username = input("아이디/이메일: ").strip()
    if not username:
        print("❌ 아이디/이메일을 입력해야 합니다.")
        sys.exit(1)
        
    password = getpass.getpass("비밀번호 (입력 시 화면에 표시되지 않음): ")
    if not password:
        print("❌ 비밀번호를 입력해야 합니다.")
        sys.exit(1)
        
    env_file = ".env"
    
    # Read existing content
    env_content = []
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            env_content = f.readlines()
            
    # Filter out existing keys for this site
    username_key = f"{site_name}_USERNAME"
    password_key = f"{site_name}_PASSWORD"
    
    new_content = []
    for line in env_content:
        if not line.startswith(f"{username_key}=") and not line.startswith(f"{password_key}="):
            new_content.append(line)
            
    # Add a newline if the last line doesn't have one
    if new_content and not new_content[-1].endswith("\n"):
        new_content[-1] += "\n"
        
    new_content.append(f"{username_key}={username}\n")
    new_content.append(f"{password_key}={password}\n")
    
    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_content)
        print(f"\n✅ [{site_name}] 자격 증명이 {env_file} 파일에 안전하게 저장되었습니다.")
        print(f"⚠️ {env_file} 파일은 절대 Git 저장소에 커밋하지 마세요 (.gitignore에 이미 포함되어 있습니다).")
            
    except Exception as e:
        print(f"\n❌ 저장 중 오류가 발생했습니다: {e}")
        logger.exception("setup_credentials CLI 실행 중 실패했습니다.")


if __name__ == "__main__":
    main()
