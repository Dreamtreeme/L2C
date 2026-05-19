import os
from dotenv import load_dotenv
from agent.utils.logger import logger
from typing import Tuple, Optional

# Load .env file
load_dotenv()

class CredentialManager:
    """
    .env 파일에서 각 채용 플랫폼의 로그인 자격 증명(ID/PW)을 읽어옵니다.
    보안을 위해 .env 파일은 .gitignore에 추가되어야 합니다.
    """

    @classmethod
    def get_credentials(cls, site_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        .env 파일에서 사이트의 자격 증명을 조회합니다.
        
        예: site_name이 'wanted'라면
        WANTED_USERNAME, WANTED_PASSWORD 환경 변수를 찾습니다.
        
        Returns:
            (username, password) 튜플 반환. 존재하지 않으면 (None, None) 반환.
        """
        prefix = site_name.upper()
        
        try:
            username = os.environ.get(f"{prefix}_USERNAME")
            password = os.environ.get(f"{prefix}_PASSWORD")
            
            if username and password:
                logger.debug(".env 파일에서 자격 증명을 성공적으로 불러왔습니다.", site=site_name)
                return username, password
            else:
                logger.warning(f".env 파일에서 자격 증명을 찾을 수 없습니다 ({prefix}_USERNAME 및 {prefix}_PASSWORD 확인 필요)", site=site_name)
                return None, None
        except Exception as e:
            logger.exception(".env 파일에서 자격 증명을 불러오는데 실패했습니다.", site=site_name, error=str(e))
            return None, None

