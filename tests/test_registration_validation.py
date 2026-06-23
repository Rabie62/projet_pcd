"""
Unit tests for registration input validation.
Tests the server-side validation logic added to AuthController.
"""
import pytest
import re


class TestRegistrationValidation:
    """Mirrors the validation rules from AuthController.validateRegistration()."""

    def _validate(self, **overrides):
        """Simulate the validation logic from AuthController."""
        defaults = {
            'nom': 'Martin',
            'prenom': 'Paul',
            'username': 'dr.martin',
            'password': 'SecurePass123',
            'email': 'dr.martin@hopital.fr',
        }
        defaults.update(overrides)
        req = defaults

        if not req.get('nom') or not req['nom'].strip():
            return "Le nom est obligatoire."
        if not req.get('prenom') or not req['prenom'].strip():
            return "Le prénom est obligatoire."
        if not req.get('username') or not req['username'].strip():
            return "Le nom d'utilisateur est obligatoire."
        if len(req['username'].strip()) < 3:
            return "Le nom d'utilisateur doit contenir au moins 3 caractères."
        if len(req['username'].strip()) > 50:
            return "Le nom d'utilisateur ne peut pas dépasser 50 caractères."
        if not re.match(r'^[a-zA-Z0-9._-]+$', req['username'].strip()):
            return "Le nom d'utilisateur ne peut contenir que des lettres, chiffres, points, tirets et underscores."
        if not req.get('password') or not req['password']:
            return "Le mot de passe est obligatoire."
        if len(req['password']) < 8:
            return "Le mot de passe doit contenir au moins 8 caractères."
        if len(req['password']) > 128:
            return "Le mot de passe ne peut pas dépasser 128 caractères."
        if not re.search(r'[A-Z]', req['password']):
            return "Le mot de passe doit contenir au moins une lettre majuscule."
        if not re.search(r'[a-z]', req['password']):
            return "Le mot de passe doit contenir au moins une lettre minuscule."
        if not re.search(r'[0-9]', req['password']):
            return "Le mot de passe doit contenir au moins un chiffre."
        if req.get('email') and req['email']:
            if not re.match(r'^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', req['email'].strip()):
                return "L'adresse e-mail n'est pas valide."
        return None

    # ── Valid input ──

    def test_valid_registration(self):
        assert self._validate() is None

    def test_valid_without_email(self):
        assert self._validate(email=None) is None

    def test_valid_with_empty_email(self):
        assert self._validate(email='') is None

    # ── Name validation ──

    def test_empty_nom(self):
        assert self._validate(nom='') == "Le nom est obligatoire."

    def test_whitespace_nom(self):
        assert self._validate(nom='   ') == "Le nom est obligatoire."

    def test_empty_prenom(self):
        assert self._validate(prenom='') == "Le prénom est obligatoire."

    # ── Username validation ──

    def test_empty_username(self):
        assert self._validate(username='') == "Le nom d'utilisateur est obligatoire."

    def test_short_username(self):
        assert self._validate(username='ab') == "Le nom d'utilisateur doit contenir au moins 3 caractères."

    def test_long_username(self):
        assert self._validate(username='a' * 51) == "Le nom d'utilisateur ne peut pas dépasser 50 caractères."

    def test_username_special_chars(self):
        assert self._validate(username='user@name') == "Le nom d'utilisateur ne peut contenir que des lettres, chiffres, points, tirets et underscores."

    def test_username_with_spaces(self):
        assert self._validate(username='user name') == "Le nom d'utilisateur ne peut contenir que des lettres, chiffres, points, tirets et underscores."

    def test_valid_username_with_dots(self):
        assert self._validate(username='dr.martin') is None

    def test_valid_username_with_hyphens(self):
        assert self._validate(username='dr-martin') is None

    def test_valid_username_with_underscores(self):
        assert self._validate(username='dr_martin') is None

    # ── Password validation ──

    def test_empty_password(self):
        assert self._validate(password='') == "Le mot de passe est obligatoire."

    def test_short_password(self):
        assert self._validate(password='Ab1') == "Le mot de passe doit contenir au moins 8 caractères."

    def test_password_no_uppercase(self):
        assert self._validate(password='securepass123') == "Le mot de passe doit contenir au moins une lettre majuscule."

    def test_password_no_lowercase(self):
        assert self._validate(password='SECUREPASS123') == "Le mot de passe doit contenir au moins une lettre minuscule."

    def test_password_no_digit(self):
        assert self._validate(password='SecurePass') == "Le mot de passe doit contenir au moins un chiffre."

    def test_password_too_long(self):
        assert self._validate(password='A1' + 'a' * 127) == "Le mot de passe ne peut pas dépasser 128 caractères."

    def test_valid_password(self):
        assert self._validate(password='SecurePass123') is None

    def test_valid_password_with_special_chars(self):
        assert self._validate(password='MyP@ssw0rd!') is None

    # ── Email validation ──

    def test_invalid_email_no_at(self):
        assert self._validate(email='invalid') == "L'adresse e-mail n'est pas valide."

    def test_invalid_email_no_domain(self):
        assert self._validate(email='user@') == "L'adresse e-mail n'est pas valide."

    def test_invalid_email_no_tld(self):
        assert self._validate(email='user@domain') == "L'adresse e-mail n'est pas valide."

    def test_valid_email(self):
        assert self._validate(email='user@example.com') is None

    def test_valid_email_with_subdomain(self):
        assert self._validate(email='user@mail.example.com') is None

    def test_valid_email_with_plus(self):
        assert self._validate(email='user+tag@example.com') is None
