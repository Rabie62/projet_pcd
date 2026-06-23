"""
Unit tests for the shared database module (data/db.py).
Tests engine/session factory singleton behavior and init_database.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestSharedDatabase:
    """Test the shared engine/session factory pattern."""

    def test_get_engine_returns_same_instance(self):
        """Calling get_engine() twice should return the same engine."""
        from data import db
        # Reset the module-level state
        db._engine = None
        db._session_factory = None

        with patch('data.db.get_settings') as mock_settings:
            mock_settings.return_value.db.url = 'mysql+pymysql://root:@localhost:3306/medical_ai'
            mock_settings.return_value.db.host = 'localhost'
            mock_settings.return_value.db.port = 3306
            mock_settings.return_value.db.name = 'medical_ai'

            with patch('data.db.create_engine') as mock_create:
                mock_engine = MagicMock()
                mock_create.return_value = mock_engine

                engine1 = db.get_engine()
                engine2 = db.get_engine()

                assert engine1 is engine2
                mock_create.assert_called_once()

    def test_get_session_factory_returns_same_instance(self):
        """Calling get_session_factory() twice should return the same factory."""
        from data import db
        db._engine = None
        db._session_factory = None

        with patch('data.db.get_settings') as mock_settings:
            mock_settings.return_value.db.url = 'mysql+pymysql://root:@localhost:3306/medical_ai'
            mock_settings.return_value.db.host = 'localhost'
            mock_settings.return_value.db.port = 3306
            mock_settings.return_value.db.name = 'medical_ai'

            with patch('data.db.create_engine') as mock_create:
                mock_create.return_value = MagicMock()

                with patch('data.db.sessionmaker') as mock_sessionmaker:
                    mock_factory = MagicMock()
                    mock_sessionmaker.return_value = mock_factory

                    factory1 = db.get_session_factory()
                    factory2 = db.get_session_factory()

                    assert factory1 is factory2
                    mock_sessionmaker.assert_called_once()

    def test_empty_url_raises_runtime_error(self):
        """An empty DB URL should raise RuntimeError with a clear message."""
        from data import db
        db._engine = None

        with patch('data.db.get_settings') as mock_settings:
            mock_settings.return_value.db.url = None

            with pytest.raises(RuntimeError, match="MySQL DATABASE_URL could not be constructed"):
                db.get_engine()

    def test_init_database_creates_tables(self):
        """init_database() should call create_all on the shared engine."""
        from data import db
        db._engine = None
        db._session_factory = None

        mock_base = MagicMock()

        with patch('data.db.get_settings') as mock_settings:
            mock_settings.return_value.db.url = 'mysql+pymysql://root:@localhost:3306/medical_ai'
            mock_settings.return_value.db.host = 'localhost'
            mock_settings.return_value.db.port = 3306
            mock_settings.return_value.db.name = 'medical_ai'

            with patch('data.db.create_engine') as mock_create:
                mock_create.return_value = MagicMock()

                with patch.dict('sys.modules', {'data.database': MagicMock(Base=mock_base)}):
                    with patch.object(db, '_engine', None, create=True):
                        db._engine = None
                        db.init_database()
                        mock_base.metadata.create_all.assert_called_once()

    def test_connection_error_propagates(self):
        """A connection failure in create_all should raise RuntimeError with context."""
        from data import db
        db._engine = None
        db._session_factory = None

        mock_base = MagicMock()
        mock_base.metadata.create_all.side_effect = Exception("Connection refused")

        mock_db_module = MagicMock()
        mock_db_module.Base = mock_base

        with patch('data.db.get_settings') as mock_settings:
            mock_settings.return_value.db.url = 'mysql+pymysql://root:@localhost:3306/medical_ai'
            mock_settings.return_value.db.host = 'localhost'
            mock_settings.return_value.db.port = 3306
            mock_settings.return_value.db.name = 'medical_ai'

            with patch('data.db.create_engine') as mock_create:
                mock_create.return_value = MagicMock()

                with patch.dict('sys.modules', {'data.database': mock_db_module}):
                    db._engine = None
                    db._session_factory = None
                    with pytest.raises(RuntimeError, match="Failed to connect to MySQL"):
                        db.init_database()
