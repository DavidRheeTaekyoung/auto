import os
import sys

def test_paths():
    print("=== 경로 테스트 ===")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"현재 스크립트 위치: {os.path.abspath(__file__)}")
    
    # 프로젝트 루트 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"프로젝트 루트: {current_dir}")
    
    # 필요한 디렉토리들 확인
    dirs_to_check = ["src", "configs", "data", "logs", "models"]
    for dir_name in dirs_to_check:
        dir_path = os.path.join(current_dir, dir_name)
        exists = os.path.exists(dir_path)
        print(f"{dir_name}: {'✓' if exists else '✗'} ({dir_path})")
    
    # Python 경로 확인
    print(f"\nPython 경로: {sys.executable}")
    print(f"Python 버전: {sys.version}")
    
    # 모듈 import 테스트
    try:
        import src.utils.config
        print("✓ config 모듈 import 성공")
    except Exception as e:
        print(f"✗ config 모듈 import 실패: {e}")
    
    try:
        import src.collectors.ohlcv_collector
        print("✓ collector 모듈 import 성공")
    except Exception as e:
        print(f"✗ collector 모듈 import 실패: {e}")

if __name__ == "__main__":
    test_paths()
