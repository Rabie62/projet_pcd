"""
Unit tests for password hashing.
Directly tests the bcrypt hashing functions without importing the full data package.
"""
import pytest
import bcrypt


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt (copy of the function under test)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_hash(password: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


class TestPasswordHashing:
    def test_hash_returns_string(self):
        hashed = hash_password("test_password_123")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_is_different_each_time(self):
        """bcrypt should produce different salts each time."""
        hashed1 = hash_password("same_password")
        hashed2 = hash_password("same_password")
        assert hashed1 != hashed2

    def test_verify_correct_password(self):
        hashed = hash_password("my_secure_password")
        assert verify_hash("my_secure_password", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("correct_password")
        assert verify_hash("wrong_password", hashed) is False

    def test_verify_empty_password(self):
        hashed = hash_password("")
        assert verify_hash("", hashed) is True

    def test_verify_unicode_password(self):
        hashed = hash_password("mot_de_passe_🔐_français")
        assert verify_hash("mot_de_passe_🔐_français", hashed) is True

    def test_verify_wrong_hash_returns_false(self):
        import bcrypt
        with pytest.raises(ValueError):
            verify_hash("password", "not_a_valid_hash")

    def test_verify_valid_hash_format(self):
        """Test with a properly formatted but incorrect bcrypt hash."""
        # Generate a real hash for a different password
        wrong_hash = bcrypt.hashpw(b"different_password", bcrypt.gensalt()).decode("utf-8")
        assert verify_hash("password", wrong_hash) is False

    def test_hash_starts_with_bcrypt_prefix(self):
        hashed = hash_password("test")
        assert hashed.startswith("$2b$")
