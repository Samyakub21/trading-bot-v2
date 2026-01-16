# =============================================================================
# POSITION RECONCILIATION - Compare Local State with Broker Positions
# =============================================================================
# Ensures local trade state matches actual broker positions

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, cast
from dataclasses import dataclass, field
from enum import Enum

from config import config
from instruments import INSTRUMENTS
from utils import send_alert, save_state, load_state

from dhanhq import dhanhq

# Initialize Dhan client (dhanhq v2.0)
dhan = dhanhq(config.CLIENT_ID, config.ACCESS_TOKEN)


# =============================================================================
# DATA CLASSES
# =============================================================================


class ReconciliationStatus(Enum):
    """Status of reconciliation check"""

    MATCHED = "matched"
    MISMATCH_LOCAL_ONLY = "local_only"  # Position in local state but not at broker
    MISMATCH_BROKER_ONLY = "broker_only"  # Position at broker but not in local state
    MISMATCH_DETAILS = "details_mismatch"  # Position exists but details differ
    ERROR = "error"


@dataclass
class BrokerPosition:
    """Represents a position from the broker"""

    security_id: str
    trading_symbol: str
    exchange_segment: str
    position_type: str  # "LONG" or "SHORT"
    quantity: int
    average_price: float
    unrealized_pnl: float
    product_type: str

    @classmethod
    def from_dhan_response(cls, data: Dict) -> "BrokerPosition":
        """Create from Dhan API V2 response"""
        # V2 API uses buyAvg/sellAvg and costPrice instead of averagePrice
        net_qty = data.get("netQty", 0)
        if net_qty > 0:
            avg_price = float(data.get("buyAvg", 0))
        else:
            avg_price = float(data.get("sellAvg", 0))

        # Fallback to costPrice if available
        if avg_price == 0:
            avg_price = float(data.get("costPrice", 0))

        return cls(
            security_id=str(data.get("securityId", "")),
            trading_symbol=data.get("tradingSymbol", ""),
            exchange_segment=data.get("exchangeSegment", ""),
            position_type="LONG" if net_qty > 0 else "SHORT",
            quantity=abs(net_qty),
            average_price=avg_price,
            unrealized_pnl=float(data.get("unrealizedProfit", 0)),
            product_type=data.get("productType", ""),
        )


@dataclass
class ReconciliationResult:
    """Result of a reconciliation check"""

    status: ReconciliationStatus
    local_state: Optional[Dict[str, Any]] = None
    broker_positions: List[BrokerPosition] = field(default_factory=list)
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    discrepancies: List[str] = field(default_factory=list)


# =============================================================================
# BROKER DATA FETCHING
# =============================================================================


def fetch_broker_positions() -> Tuple[bool, List[BrokerPosition]]:
    """
    Fetch all current positions from the broker.

    Returns:
        Tuple of (success, list of positions)
    """
    try:
        response = dhan.get_positions()

        if response.get("status") == "failure":
            logging.error(
                f"Failed to fetch positions: {response.get('remarks', 'Unknown error')}"
            )
            return False, []

        positions = []
        for pos_data in response.get("data", []):
            # Only include positions with non-zero quantity
            if pos_data.get("netQty", 0) != 0:
                positions.append(BrokerPosition.from_dhan_response(pos_data))

        logging.info(f"📊 Fetched {len(positions)} open position(s) from broker")
        return True, positions

    except Exception as e:
        logging.error(f"Error fetching broker positions: {e}")
        return False, []


def fetch_broker_holdings() -> Tuple[bool, List[Dict]]:
    """
    Fetch holdings (for equity, if needed).
    """
    try:
        response = dhan.get_holdings()

        if response.get("status") == "failure":
            return False, []

        return True, response.get("data", [])

    except Exception as e:
        logging.error(f"Error fetching holdings: {e}")
        return False, []


def fetch_today_orders() -> Tuple[bool, List[Dict]]:
    """
    Fetch today's orders for verification.
    """
    try:
        response = dhan.get_order_list()

        if response.get("status") == "failure":
            return False, []

        return True, response.get("data", [])

    except Exception as e:
        logging.error(f"Error fetching orders: {e}")
        return False, []


def fetch_today_trades() -> Tuple[bool, List[Dict]]:
    """
    Fetch today's executed trades.
    """
    try:
        response = dhan.get_trade_book()

        if response.get("status") == "failure":
            return False, []

        return True, response.get("data", [])

    except Exception as e:
        logging.error(f"Error fetching trades: {e}")
        return False, []


# =============================================================================
# RECONCILIATION LOGIC
# =============================================================================


def reconcile_positions(active_trade: Dict[str, Any]) -> ReconciliationResult:
    """
    Compare local trade state with broker positions.

    Args:
        active_trade: The local active_trade state dictionary

    Returns:
        ReconciliationResult with status and details
    """
    result = ReconciliationResult(
        status=ReconciliationStatus.MATCHED,
        local_state=active_trade.copy() if active_trade else None,
    )

    # Fetch broker positions
    success, broker_positions = fetch_broker_positions()

    if not success:
        result.status = ReconciliationStatus.ERROR
        result.message = "Failed to fetch broker positions"
        return result

    result.broker_positions = broker_positions

    # Check local state
    local_has_position = active_trade.get("status", False)
    local_option_id = (
        str(active_trade.get("option_id", "")) if local_has_position else None
    )

    # Find matching broker position
    matching_broker_position = None
    if local_option_id:
        for pos in broker_positions:
            if pos.security_id == local_option_id:
                matching_broker_position = pos
                break

    # Case 1: Local shows position, broker doesn't have it
    if local_has_position and not matching_broker_position:
        result.status = ReconciliationStatus.MISMATCH_LOCAL_ONLY
        result.message = f"Local state shows position in {active_trade.get('instrument')} but broker has no matching position"
        result.discrepancies.append(
            f"Local option_id {local_option_id} not found at broker"
        )

        logging.error(f"🚨 RECONCILIATION MISMATCH: {result.message}")
        send_alert(
            f"🚨 **POSITION MISMATCH**\n{result.message}\n\nManual verification required!"
        )

        return result

    # Case 2: Local shows no position, broker has positions
    if not local_has_position and broker_positions:
        # Check if any of the broker positions look like our trades
        our_positions = []
        for pos in broker_positions:
            # Check if it's an options position on one of our instruments
            for inst_key, inst_config in INSTRUMENTS.items():
                if cast(str, inst_config.get("exchange_segment_str")) in pos.exchange_segment:
                    our_positions.append(pos)
                    break

        if our_positions:
            result.status = ReconciliationStatus.MISMATCH_BROKER_ONLY
            result.message = f"Broker has {len(our_positions)} position(s) but local state shows no active trade"
            result.discrepancies.append(
                f"Untracked positions: {[p.trading_symbol for p in our_positions]}"
            )

            logging.warning(f"⚠️ RECONCILIATION WARNING: {result.message}")
            send_alert(
                f"⚠️ **UNTRACKED POSITION**\n{result.message}\n\nCheck if bot missed a trade entry!"
            )

            return result

    # Case 3: Both have position - verify details match
    if local_has_position and matching_broker_position:
        local_lot_size = active_trade.get("lot_size", 0)

        if matching_broker_position.quantity != local_lot_size:
            result.status = ReconciliationStatus.MISMATCH_DETAILS
            result.message = f"Quantity mismatch: Local={local_lot_size}, Broker={matching_broker_position.quantity}"
            result.discrepancies.append(result.message)

            logging.warning(f"⚠️ RECONCILIATION WARNING: {result.message}")
        else:
            result.message = "Position reconciled successfully"
            logging.info(
                f"✅ Position reconciled: {active_trade.get('instrument')} @ ₹{matching_broker_position.average_price}"
            )

    # Case 4: Both show no position - all clear
    if not local_has_position and not broker_positions:
        result.message = "No positions - reconciliation passed"
        logging.info("✅ No positions - reconciliation passed")

    return result


def auto_fix_mismatch(
    result: ReconciliationResult, active_trade: Dict[str, Any]
) -> bool:
    """
    Attempt to auto-fix a position mismatch.

    Args:
        result: The reconciliation result
        active_trade: The local active_trade dictionary to update

    Returns:
        True if fix was applied, False otherwise
    """
    if result.status == ReconciliationStatus.MATCHED:
        return True

    if result.status == ReconciliationStatus.MISMATCH_LOCAL_ONLY:
        # Local shows position but broker doesn't - likely position was closed manually
        logging.warning(
            "⚠️ Auto-fixing: Clearing local state (position appears closed at broker)"
        )

        # Clear local state
        active_trade["status"] = False
        active_trade["type"] = None
        active_trade["option_id"] = None
        active_trade["instrument"] = None
        save_state(active_trade)

        send_alert(
            "🔧 **AUTO-FIX APPLIED**\n"
            "Local state cleared - position was closed at broker.\n"
            "If this is incorrect, please check manually."
        )

        return True

    if result.status == ReconciliationStatus.MISMATCH_BROKER_ONLY:
        # Broker has position but local doesn't - need manual intervention
        logging.error("❌ Cannot auto-fix: Broker has untracked position")
        send_alert(
            "🚨 **MANUAL INTERVENTION REQUIRED**\n"
            "Broker has position(s) not tracked locally.\n"
            "Please verify and either:\n"
            "1. Update local state manually\n"
            "2. Close the position at broker"
        )
        return False

    if result.status == ReconciliationStatus.MISMATCH_DETAILS:
        # Details mismatch - update local to match broker
        if result.broker_positions:
            broker_pos = result.broker_positions[0]

            logging.warning(
                f"⚠️ Auto-fixing: Updating local quantity to match broker ({broker_pos.quantity})"
            )
            active_trade["lot_size"] = broker_pos.quantity
            save_state(active_trade)

            send_alert(
                f"🔧 **AUTO-FIX APPLIED**\n"
                f"Updated local lot_size to {broker_pos.quantity} to match broker."
            )

            return True

    return False


# =============================================================================
# SCHEDULED RECONCILIATION
# =============================================================================


def run_periodic_reconciliation(
    active_trade: Dict[str, Any], interval_seconds: int = 300
):
    """
    Run reconciliation periodically.

    Args:
        active_trade: The active_trade state dictionary
        interval_seconds: How often to reconcile (default 5 minutes)
    """
    import threading

    def reconcile_loop():
        while True:
            try:
                result = reconcile_positions(active_trade)

                if result.status != ReconciliationStatus.MATCHED:
                    logging.warning(f"⚠️ Reconciliation issue: {result.message}")

                    # Attempt auto-fix for certain mismatches
                    if result.status in [
                        ReconciliationStatus.MISMATCH_LOCAL_ONLY,
                        ReconciliationStatus.MISMATCH_DETAILS,
                    ]:
                        auto_fix_mismatch(result, active_trade)

            except Exception as e:
                logging.error(f"Reconciliation error: {e}")

            time.sleep(interval_seconds)

    thread = threading.Thread(
        target=reconcile_loop, daemon=True, name="ReconciliationThread"
    )
    thread.start()
    logging.info(f"🔄 Position reconciliation started (every {interval_seconds}s)")
    return thread


# =============================================================================
# STARTUP RECONCILIATION
# =============================================================================


def reconcile_on_startup(active_trade: Dict[str, Any]) -> bool:
    """
    Run reconciliation at bot startup.

    Args:
        active_trade: The active_trade state dictionary

    Returns:
        True if reconciliation passed (or was fixed), False if manual intervention needed
    """
    logging.info("=" * 60)
    logging.info("🔍 STARTUP POSITION RECONCILIATION")
    logging.info("=" * 60)

    result = reconcile_positions(active_trade)

    if result.status == ReconciliationStatus.MATCHED:
        logging.info("✅ Reconciliation passed - positions match")
        return True

    if result.status == ReconciliationStatus.ERROR:
        logging.error(f"❌ Reconciliation error: {result.message}")
        # Proceed with caution but don't block startup
        return True

    # Attempt auto-fix
    fixed = auto_fix_mismatch(result, active_trade)

    if not fixed:
        logging.error("❌ Reconciliation failed - manual intervention required")
        logging.error(f"   Status: {result.status.value}")
        logging.error(f"   Message: {result.message}")
        for disc in result.discrepancies:
            logging.error(f"   Discrepancy: {disc}")

        # Don't block startup, but warn loudly
        send_alert(
            "🚨 **STARTUP RECONCILIATION FAILED**\n"
            f"Status: {result.status.value}\n"
            f"Message: {result.message}\n"
            "Bot will continue but please verify positions manually!"
        )

    return fixed


# =============================================================================
# POSITION VERIFICATION
# =============================================================================


def verify_trade_entry(
    order_id: str, expected_security_id: str, expected_qty: int
) -> Tuple[bool, str]:
    """
    Verify a trade entry was successful by checking broker position.

    Args:
        order_id: The order ID to verify
        expected_security_id: Expected security ID
        expected_qty: Expected quantity

    Returns:
        Tuple of (success, message)
    """
    # Wait for order to settle
    time.sleep(2)

    # Fetch positions
    success, positions = fetch_broker_positions()

    if not success:
        return False, "Failed to fetch positions for verification"

    for pos in positions:
        if pos.security_id == expected_security_id:
            if pos.quantity >= expected_qty:
                return True, f"Entry verified: {pos.trading_symbol} qty={pos.quantity}"
            else:
                return (
                    False,
                    f"Quantity mismatch: expected {expected_qty}, got {pos.quantity}",
                )

    return False, f"Position not found for security_id {expected_security_id}"


def verify_trade_exit(expected_security_id: str) -> Tuple[bool, str]:
    """
    Verify a trade exit was successful (position should be closed).

    Args:
        expected_security_id: Security ID that should be closed

    Returns:
        Tuple of (success, message)
    """
    # Wait for order to settle
    time.sleep(2)

    # Fetch positions
    success, positions = fetch_broker_positions()

    if not success:
        return False, "Failed to fetch positions for verification"

    for pos in positions:
        if pos.security_id == expected_security_id:
            return False, f"Position still open: qty={pos.quantity}"

    return True, "Exit verified: position closed"


# =============================================================================
# REPORTING
# =============================================================================


def generate_reconciliation_report(active_trade: Dict[str, Any]) -> str:
    """
    Generate a detailed reconciliation report.
    """
    result = reconcile_positions(active_trade)

    report_lines = [
        "=" * 60,
        "📊 POSITION RECONCILIATION REPORT",
        f"🕐 Time: {result.timestamp}",
        "=" * 60,
        "",
        "--- LOCAL STATE ---",
    ]

    if result.local_state and result.local_state.get("status"):
        report_lines.extend(
            [
                f"Status: Active",
                f"Instrument: {result.local_state.get('instrument')}",
                f"Type: {result.local_state.get('type')}",
                f"Option ID: {result.local_state.get('option_id')}",
                f"Entry: ₹{result.local_state.get('option_entry', 0)}",
                f"Current SL: {result.local_state.get('sl', 0)}",
                f"Lot Size: {result.local_state.get('lot_size', 0)}",
            ]
        )
    else:
        report_lines.append("Status: No active position")

    report_lines.extend(["", "--- BROKER POSITIONS ---"])

    if result.broker_positions:
        for pos in result.broker_positions:
            report_lines.extend(
                [
                    f"Symbol: {pos.trading_symbol}",
                    f"  Security ID: {pos.security_id}",
                    f"  Type: {pos.position_type}",
                    f"  Quantity: {pos.quantity}",
                    f"  Avg Price: ₹{pos.average_price:.2f}",
                    f"  Unrealized P&L: ₹{pos.unrealized_pnl:.2f}",
                    "",
                ]
            )
    else:
        report_lines.append("No open positions")

    report_lines.extend(
        [
            "",
            "--- RECONCILIATION RESULT ---",
            f"Status: {result.status.value.upper()}",
            f"Message: {result.message}",
        ]
    )

    if result.discrepancies:
        report_lines.append("Discrepancies:")
        for disc in result.discrepancies:
            report_lines.append(f"  - {disc}")

    report_lines.append("=" * 60)

    return "\n".join(report_lines)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Position Reconciliation CLI")
    parser.add_argument("--check", action="store_true", help="Run reconciliation check")
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report"
    )
    parser.add_argument(
        "--positions", action="store_true", help="List broker positions"
    )
    parser.add_argument("--orders", action="store_true", help="List today's orders")
    parser.add_argument("--trades", action="store_true", help="List today's trades")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    active_trade = load_state()

    if args.check:
        result = reconcile_positions(active_trade)
        print(f"\nStatus: {result.status.value}")
        print(f"Message: {result.message}")
        if result.discrepancies:
            print("Discrepancies:")
            for d in result.discrepancies:
                print(f"  - {d}")

    if args.report:
        report = generate_reconciliation_report(active_trade)
        print(report)

    if args.positions:
        success, positions = fetch_broker_positions()
        if success:
            print(f"\n📊 Open Positions ({len(positions)}):")
            for pos in positions:
                print(
                    f"  {pos.trading_symbol}: {pos.position_type} {pos.quantity} @ ₹{pos.average_price:.2f}"
                )
        else:
            print("Failed to fetch positions")

    if args.orders:
        success, orders = fetch_today_orders()
        if success:
            print(f"\n📋 Today's Orders ({len(orders)}):")
            for order in orders:
                print(
                    f"  {order.get('tradingSymbol')}: {order.get('transactionType')} {order.get('quantity')} - {order.get('orderStatus')}"
                )

    if args.trades:
        success, trades = fetch_today_trades()
        if success:
            print(f"\n📋 Today's Trades ({len(trades)}):")
            for trade in trades:
                print(
                    f"  {trade.get('tradingSymbol')}: {trade.get('transactionType')} {trade.get('tradedQuantity')} @ ₹{trade.get('tradedPrice')}"
                )
