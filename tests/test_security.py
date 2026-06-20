"""
Tests for security middleware and utilities.
Run: python -m pytest tests/test_security.py -v
"""
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware.security import (
    CSP_POLICY,
    build_csp_header,
    CSRFProtection,
    sanitize_input,
    sanitize_filename,
    validate_mime_type,
    secure_error_response,
)


class TestCSPPolicy:
    """Test Content Security Policy configuration."""

    def test_csp_is_dict(self):
        assert isinstance(CSP_POLICY, dict)

    def test_csp_has_required_directives(self):
        required = ["default-src", "script-src", "style-src", "img-src", "connect-src"]
        for directive in required:
            assert directive in CSP_POLICY, f"Missing CSP directive: {directive}"

    def test_csp_default_src_strict(self):
        assert "'self'" in CSP_POLICY["default-src"]

    def test_csp_connect_src(self):
        connect = CSP_POLICY["connect-src"]
        assert "'self'" in connect

    def test_csp_frame_ancestors(self):
        assert "'none'" in CSP_POLICY.get("frame-ancestors", [])


class TestBuildCSPHeader:
    """Test CSP header builder."""

    def test_returns_string(self):
        header = build_csp_header()
        assert isinstance(header, str)

    def test_contains_semicolons(self):
        header = build_csp_header()
        assert ";" in header

    def test_contains_default_src(self):
        header = build_csp_header()
        assert "default-src" in header


class TestCSRFProtection:
    """Test CSRF token generation and validation."""

    def test_generate_token_returns_string(self):
        csrf = CSRFProtection(secret_key="test-secret-key")
        token = csrf.generate_token("session_123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token_valid(self):
        csrf = CSRFProtection(secret_key="test-secret-key")
        token = csrf.generate_token("session_123")
        assert csrf.validate_token(token, "session_123") is True

    def test_validate_token_wrong_session(self):
        csrf = CSRFProtection(secret_key="test-secret-key")
        token = csrf.generate_token("session_123")
        assert csrf.validate_token(token, "session_456") is False

    def test_validate_token_expired(self):
        csrf = CSRFProtection(secret_key="test-secret-key")
        token = csrf.generate_token("session_123")
        assert csrf.validate_token(token, "session_123", max_age=0) is False

    def test_validate_token_empty(self):
        csrf = CSRFProtection(secret_key="test-secret-key")
        assert csrf.validate_token("", "session_123") is False


class TestSanitizeInput:
    """Test HTML input sanitization."""

    def test_plain_text_unchanged(self):
        assert sanitize_input("hello world") == "hello world"

    def test_html_tags_encoded(self):
        result = sanitize_input("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;" in result

    def test_ampersand_encoded(self):
        result = sanitize_input("a & b")
        assert "&amp;" in result

    def test_quotes_encoded(self):
        result = sanitize_input('"hello"')
        assert "&quot;" in result


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_normal_filename(self):
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_path_traversal_stripped(self):
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_backslash_stripped(self):
        result = sanitize_filename("..\\windows\\system32")
        assert "\\" not in result
        assert ".." not in result

    def test_empty_returns_unnamed(self):
        assert sanitize_filename("") == "unnamed"

    def test_none_returns_unnamed(self):
        assert sanitize_filename(None) == "unnamed"


class TestValidateMimeType:
    """Test MIME type validation."""

    def test_valid_png(self):
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert validate_mime_type(png_header, "image.png") is True

    def test_valid_jpeg(self):
        jpeg_header = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert validate_mime_type(jpeg_header, "photo.jpg") is True

    def test_fake_png_rejected(self):
        assert validate_mime_type(b"not a png at all", "test.png") is False

    def test_non_image_extension(self):
        assert validate_mime_type(b"random", "file.exe") is False

    def test_medical_nii_allowed(self):
        assert validate_mime_type(b"NIFTI", "scan.nii") is True

    def test_medical_dcm_allowed(self):
        assert validate_mime_type(b"DICM", "scan.dcm") is True

    def test_empty_content_rejected(self):
        assert validate_mime_type(b"", "image.png") is False

    def test_none_content_rejected(self):
        assert validate_mime_type(None, "image.png") is False


class TestSecureErrorResponse:
    """Test secure error response generation."""

    def test_returns_json_response(self):
        resp = secure_error_response(404)
        assert resp.status_code == 404

    def test_does_not_leak_details(self):
        resp = secure_error_response(500, "Internal DB password: abc123")
        body = json.loads(resp.body)
        assert "abc123" not in body.get("response", "")
        assert "password" not in body.get("response", "").lower()

    def test_429_too_many_requests(self):
        resp = secure_error_response(429)
        body = json.loads(resp.body)
        assert body["status"] == "error"
        assert "429" in str(resp.status_code)

    def test_500_generic_message(self):
        resp = secure_error_response(500)
        body = json.loads(resp.body)
        assert "internal" in body["error"].lower()

    def test_detail_is_logged_not_returned(self):
        resp = secure_error_response(403, "secret debug info")
        body = json.loads(resp.body)
        assert "secret" not in body["error"]
        assert "denied" in body["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
