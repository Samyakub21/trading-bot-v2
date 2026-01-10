#!/usr/bin/env python3
"""
Password Hash Generator Utility for Trading Bot Dashboard

This script generates bcrypt password hashes for use in dashboard.py
Run this to create secure password hashes for new users.

Usage:
    python generate_password_hash.py

Requirements:
    pip install bcrypt
"""

import re
import sys
import getpass
from typing import Tuple, List

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    print("Error: bcrypt not installed. Run: pip install bcrypt")
    sys.exit(1)


# Password Policy (must match dashboard.py)
PASSWORD_POLICY = {
    "min_length": 12,
    "require_uppercase": True,
    "require_lowercase": True,
    "require_digit": True,
    "require_special": True,
    "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?"
}


def validate_password(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password against security policy.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: List[str] = []
    
    if len(password) < PASSWORD_POLICY["min_length"]:
        errors.append(f"❌ Password must be at least {PASSWORD_POLICY['min_length']} characters (got {len(password)})")
    else:
        errors.append(f"✅ Length: {len(password)} characters")
    
    if PASSWORD_POLICY["require_uppercase"]:
        if re.search(r'[A-Z]', password):
            errors.append("✅ Contains uppercase letter")
        else:
            errors.append("❌ Must contain at least one uppercase letter (A-Z)")
    
    if PASSWORD_POLICY["require_lowercase"]:
        if re.search(r'[a-z]', password):
            errors.append("✅ Contains lowercase letter")
        else:
            errors.append("❌ Must contain at least one lowercase letter (a-z)")
    
    if PASSWORD_POLICY["require_digit"]:
        if re.search(r'\d', password):
            errors.append("✅ Contains digit")
        else:
            errors.append("❌ Must contain at least one digit (0-9)")
    
    if PASSWORD_POLICY["require_special"]:
        special_pattern = f"[{re.escape(PASSWORD_POLICY['special_chars'])}]"
        if re.search(special_pattern, password):
            errors.append("✅ Contains special character")
        else:
            errors.append(f"❌ Must contain at least one special character: {PASSWORD_POLICY['special_chars']}")
    
    is_valid = not any("❌" in e for e in errors)
    return is_valid, errors


def hash_password(password: str) -> str:
    """Generate bcrypt hash for password"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def main():
    print("=" * 60)
    print("🔐 Trading Bot Dashboard - Password Hash Generator")
    print("=" * 60)
    print()
    print("Password Requirements:")
    print(f"  • Minimum {PASSWORD_POLICY['min_length']} characters")
    print("  • At least one uppercase letter (A-Z)")
    print("  • At least one lowercase letter (a-z)")
    print("  • At least one digit (0-9)")
    print(f"  • At least one special character: {PASSWORD_POLICY['special_chars']}")
    print()
    print("-" * 60)
    
    while True:
        # Get username
        username = input("\nEnter username (or 'quit' to exit): ").strip()
        if username.lower() == 'quit':
            break
        
        if not username:
            print("❌ Username cannot be empty")
            continue
        
        # Get password
        password = getpass.getpass("Enter password: ")
        
        # Validate password
        is_valid, messages = validate_password(password)
        
        print("\nPassword Validation:")
        for msg in messages:
            print(f"  {msg}")
        
        if not is_valid:
            print("\n❌ Password does not meet requirements. Please try again.")
            continue
        
        # Confirm password
        password_confirm = getpass.getpass("Confirm password: ")
        
        if password != password_confirm:
            print("\n❌ Passwords do not match. Please try again.")
            continue
        
        # Generate hash
        password_hash = hash_password(password)
        
        # Verify the hash works
        if verify_password(password, password_hash):
            print("\n" + "=" * 60)
            print("✅ Password hash generated successfully!")
            print("=" * 60)
            print(f"\nUsername: {username}")
            print(f"Hash: {password_hash}")
            print()
            print("Add this to DASHBOARD_USERS in dashboard.py:")
            print("-" * 60)
            print(f'    "{username}": "{password_hash}",')
            print("-" * 60)
        else:
            print("\n❌ Hash verification failed. Please try again.")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
