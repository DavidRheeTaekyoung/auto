from loguru import logger
import os, sys

def setup_logger(log_dir=None, level="INFO"):
    if log_dir is None:
        # 현재 스크립트 위치 기준으로 상대 경로 계산
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        log_dir = os.path.join(project_root, "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=level)
    logger.add(f"{log_dir}/btc_{'{time}'.replace(':','-')}.log",
               level=level, rotation="1 day", retention="15 days")
    return logger
