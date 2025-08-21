import os
import sys
import numpy as np
import pandas as pd

def test_system():
    print("=== 비트코인 거래 시스템 테스트 ===")
    
    # 1. 설정 로드 테스트
    try:
        from src.utils.config import load_config
        cfg = load_config()
        print("✓ 설정 로드 성공")
        print(f"  - 거래소: {cfg['data']['exchange']}")
        print(f"  - 심볼: {cfg['data']['symbol_internal']}")
        print(f"  - 최대 레버리지: {cfg['trade']['max_leverage']}x")
    except Exception as e:
        print(f"✗ 설정 로드 실패: {e}")
        return False
    
    # 2. 로거 테스트
    try:
        from src.utils.logger import setup_logger
        log = setup_logger(level="INFO")
        log.info("로거 테스트 성공")
        print("✓ 로거 설정 성공")
    except Exception as e:
        print(f"✗ 로거 설정 실패: {e}")
        return False
    
    # 3. 특징 생성 테스트
    try:
        from src.features.feature_maker import enrich, make_dataset
        
        # 테스트 데이터 생성
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        prices = 42000 + np.cumsum(np.random.randn(1000) * 0.001)
        
        test_df = pd.DataFrame({
            'open_time': dates,
            'open': prices + np.random.randn(1000) * 10,
            'high': prices + np.random.randn(1000) * 20,
            'low': prices - np.random.randn(1000) * 20,
            'close': prices,
            'volume': np.random.uniform(100, 200, 1000)
        })
        
        # 특징 생성
        feat = enrich(test_df)
        X, Y = make_dataset(feat, input_window=100, horizons=(60, 240))
        
        print("✓ 특징 생성 성공")
        print(f"  - 입력 데이터: {X.shape}")
        print(f"  - 타겟 데이터: {Y.shape}")
        
    except Exception as e:
        print(f"✗ 특징 생성 실패: {e}")
        return False
    
    # 4. 거래 로직 테스트
    try:
        from src.traders.sizing import confidence_from_quantiles, leverage_from_confidence
        
        # 테스트 예측값
        q10, q50, q90 = -0.02, 0.01, 0.04
        
        # 확신도 및 레버리지 계산
        conf, direction = confidence_from_quantiles(q10, q50, q90)
        lev = leverage_from_confidence(conf, max_leverage=25, base=5)
        
        print("✓ 거래 로직 테스트 성공")
        print(f"  - 확신도: {conf:.3f}")
        print(f"  - 방향: {'롱' if direction > 0 else '숏'}")
        print(f"  - 레버리지: {lev:.1f}x")
        
    except Exception as e:
        print(f"✗ 거래 로직 테스트 실패: {e}")
        return False
    
    # 5. 리스크 관리 테스트
    try:
        from src.traders.risk import stop_price_for_account_risk, TrailingManager
        
        # 손절가 계산 테스트
        entry_price = 42000
        qty = 1.0
        cash = 10000
        stop = stop_price_for_account_risk(entry_price, qty, cash, is_long=True, cash_risk_pct=0.10)
        
        # 트레일링 테스트
        trailer = TrailingManager(0.30)
        trailer.update(1000)  # 1000 USDT 이익
        trail_stop = trailer.trailing_stop_price(entry_price, qty, is_long=True)
        
        print("✓ 리스크 관리 테스트 성공")
        print(f"  - 손절가: {stop:.2f}")
        print(f"  - 트레일링 스탑: {trail_stop:.2f}")
        
    except Exception as e:
        print(f"✗ 리스크 관리 테스트 실패: {e}")
        return False
    
    print("\n🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
    return True

if __name__ == "__main__":
    success = test_system()
    if not success:
        print("\n❌ 일부 테스트가 실패했습니다.")
        sys.exit(1)
