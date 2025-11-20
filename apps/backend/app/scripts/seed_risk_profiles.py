# app/scripts/seed_risk_profiles.py

from app.db.database import SessionLocal
from app.db import models

def main():
    db = SessionLocal()
    try:
        existing = db.query(models.RiskSettingsSymbol).count()
        if existing > 0:
            print(f"Risk settings already seeded: {existing} rows.")
            return

        profiles = []

        # Conservative profile
        profiles.append(
            models.RiskSettingsSymbol(
                symbol="BTC",
                profile_name="CONSERVATIVE",
                max_leverage=3,
                max_daily_loss_pct=2.0,
                base_position_risk_pct=0.25,
                max_position_risk_pct=0.5,
                min_sl_atr_mult=1.5,
                max_sl_atr_mult=3.0,
                min_tp_atr_mult=2.0,
                max_tp_atr_mult=4.0,
                ewma_alpha=0.05,
                is_active=True,
            )
        )
        profiles.append(
            models.RiskSettingsSymbol(
                symbol="ETH",
                profile_name="CONSERVATIVE",
                max_leverage=3,
                max_daily_loss_pct=2.0,
                base_position_risk_pct=0.25,
                max_position_risk_pct=0.5,
                min_sl_atr_mult=1.5,
                max_sl_atr_mult=3.0,
                min_tp_atr_mult=2.0,
                max_tp_atr_mult=4.0,
                ewma_alpha=0.05,
                is_active=True,
            )
        )

        # Balanced profile
        profiles.append(
            models.RiskSettingsSymbol(
                symbol="BTC",
                profile_name="BALANCED",
                max_leverage=5,
                max_daily_loss_pct=3.0,
                base_position_risk_pct=0.5,
                max_position_risk_pct=1.0,
                min_sl_atr_mult=1.2,
                max_sl_atr_mult=2.5,
                min_tp_atr_mult=2.0,
                max_tp_atr_mult=5.0,
                ewma_alpha=0.08,
                is_active=True,
            )
        )

        profiles.append(
            models.RiskSettingsSymbol(
                symbol="ETH",
                profile_name="BALANCED",
                max_leverage=5,
                max_daily_loss_pct=3.0,
                base_position_risk_pct=0.5,
                max_position_risk_pct=1.0,
                min_sl_atr_mult=1.2,
                max_sl_atr_mult=2.5,
                min_tp_atr_mult=2.0,
                max_tp_atr_mult=5.0,
                ewma_alpha=0.08,
                is_active=True,
            )
        )

        # Aggressive profile
        profiles.append(
            models.RiskSettingsSymbol(
                symbol="BTC",
                profile_name="AGGRESSIVE",
                max_leverage=10,
                max_daily_loss_pct=5.0,
                base_position_risk_pct=1.0,
                max_position_risk_pct=2.0,
                min_sl_atr_mult=0.8,
                max_sl_atr_mult=2.0,
                min_tp_atr_mult=1.5,
                max_tp_atr_mult=6.0,
                ewma_alpha=0.12,
                is_active=True,
            )
        )

        profiles.append(
            models.RiskSettingsSymbol(
                symbol="ETH",
                profile_name="AGGRESSIVE",
                max_leverage=10,
                max_daily_loss_pct=5.0,
                base_position_risk_pct=1.0,
                max_position_risk_pct=2.0,
                min_sl_atr_mult=0.8,
                max_sl_atr_mult=2.0,
                min_tp_atr_mult=1.5,
                max_tp_atr_mult=6.0,
                ewma_alpha=0.12,
                is_active=True,
            )
        )

        db.add_all(profiles)
        db.commit()
        print(f"Seeded {len(profiles)} risk profiles.")

    finally:
        db.close()

if __name__ == "__main__":
    main()
