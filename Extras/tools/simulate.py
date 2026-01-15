import time
import random

# --- MOCK VARIABLES ---
active_trade = {
    "status": True,
    "type": "BUY",
    "entry_price": 6000,
    "initial_sl": 5980,  # 20 points risk
    "current_sl_level": 5980,
    "step_level": 0,
    "order_id": "12345",
}

print(f"--- SIMULATION STARTED ---")
print(
    f"Entry: {active_trade['entry_price']} | SL: {active_trade['initial_sl']} | Risk: 20 pts"
)
print("-" * 50)

# Simulate 50 minutes of price movement
current_price = 6000
for i in range(1, 51):
    # Randomly move price (Simulating volatility)
    move = random.choice([-5, -2, 0, 2, 5, 8, 10])
    current_price += move

    # --- YOUR EXACT MANAGER LOGIC ---
    ltp = current_price

    # 1. Calc Risk (R)
    risk_unit = abs(active_trade["entry_price"] - active_trade["initial_sl"])

    # Calc R-Multiple (guard against zero risk_unit)
    if active_trade["type"] == "BUY":
        profit_points = ltp - active_trade["entry_price"]
    else:
        profit_points = active_trade["entry_price"] - ltp

    if risk_unit == 0:
        print("Warning: risk_unit is zero; skipping r-multiple calculations")
        r_multiple = 0.0
    else:
        r_multiple = profit_points / risk_unit

    print(
        f"Time: {i}m | Price: {ltp} | Profit: {profit_points} pts ({r_multiple:.1f}R) | SL: {active_trade['current_sl_level']}"
    )

    # 2. STEP LADDER
    if r_multiple >= 5.0:
        print(f">>> 🚀 TARGET HIT (1:5)! Exiting at {ltp}")
        break

    # Trailing Logic
    # reset per-iteration lock variable
    lock_r = None
    new_sl = None
    if r_multiple >= 4.0 and active_trade["step_level"] < 4:
        print("   >>> 🔒 LOCKING 3R (Step 4)")
        lock_r = 3.0
        active_trade["step_level"] = 4
    elif r_multiple >= 3.0 and active_trade["step_level"] < 3:
        print("   >>> 🔒 LOCKING 2R (Step 3)")
        lock_r = 2.0
        active_trade["step_level"] = 3
    elif r_multiple >= 2.0 and active_trade["step_level"] < 2:
        print("   >>> 🔒 LOCKING 1R (Step 2)")
        lock_r = 1.0
        active_trade["step_level"] = 2

    # Execute SL Move (only tighten stop, don't move it backwards)
    if lock_r is not None and active_trade["step_level"] > 0:
        if active_trade["type"] == "BUY":
            candidate = active_trade["entry_price"] + (lock_r * risk_unit)
            if candidate > active_trade["current_sl_level"]:
                active_trade["current_sl_level"] = candidate
        else:
            candidate = active_trade["entry_price"] - (lock_r * risk_unit)
            if candidate < active_trade["current_sl_level"]:
                active_trade["current_sl_level"] = candidate

    # CHECK IF SL HIT
    if ltp <= active_trade["current_sl_level"]:
        print(f">>> 🛑 SL HIT at {ltp}. Trade Closed.")
        break

    time.sleep(0.1)  # Fast forward
