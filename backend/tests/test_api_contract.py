"""Focused API-boundary coverage; no lifespan, retrieval, or provider calls occur."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.services.api_config import LOCAL_DEVELOPMENT_ORIGINS, CorsSettings, configure_cors


class ApiContractTests(unittest.TestCase):
    def test_openapi_keeps_stable_one_shot_paths_and_safe_error_schema(self) -> None:
        schema = app.openapi()
        self.assertIn("/query-text", schema["paths"])
        self.assertIn("/query-voice", schema["paths"])
        self.assertIn("/health", schema["paths"])
        self.assertIn("/ready", schema["paths"])
        self.assertIn("/diagnostics", schema["paths"])
        text_operation = schema["paths"]["/query-text"]["post"]
        voice_operation = schema["paths"]["/query-voice"]["post"]
        self.assertIn("TextQueryRequest", str(text_operation))
        self.assertIn("HarnessErrorResponse", str(text_operation["responses"]["422"]))
        self.assertIn("HarnessErrorResponse", str(voice_operation["responses"]["415"]))
        body_ref = voice_operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[-1]
        self.assertIn("audio", schema["components"]["schemas"][body_ref]["properties"])

    def test_openapi_examples_do_not_include_environment_credentials(self) -> None:
        schema_text = str(app.openapi())
        self.assertNotIn("QDRANT_API_KEY", schema_text)
        self.assertNotIn("SARVAM_API_KEY", schema_text)
        self.assertNotIn("DIAGNOSTICS_API_KEY", schema_text)

    def test_cors_accepts_configured_origin_and_omits_unconfigured_origin(self) -> None:
        cors_app = FastAPI()

        @cors_app.get("/probe")
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        configure_cors(cors_app, CorsSettings(("https://frontend.example",)))
        with TestClient(cors_app) as client:
            allowed = client.options(
                "/probe",
                headers={"Origin": "https://frontend.example", "Access-Control-Request-Method": "POST"},
            )
            denied = client.options(
                "/probe",
                headers={"Origin": "https://unexpected.example", "Access-Control-Request-Method": "POST"},
            )
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "https://frontend.example")
        self.assertNotIn("access-control-allow-origin", denied.headers)

    def test_cors_environment_parser_is_explicit_and_never_wildcard(self) -> None:
        with patch.dict(os.environ, {"FRONTEND_ORIGINS": "https://ui.example, http://localhost:5173/"}, clear=False):
            settings = CorsSettings.from_environment()
        self.assertEqual(settings.origins, ("https://ui.example", "http://localhost:5173"))
        with patch.dict(os.environ, {"FRONTEND_ORIGINS": "*"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "must not contain"):
                CorsSettings.from_environment()
        with patch.dict(os.environ, {"FRONTEND_ORIGINS": ""}, clear=False):
            self.assertEqual(CorsSettings.from_environment().origins, LOCAL_DEVELOPMENT_ORIGINS)

    def test_env_example_contains_placeholders_not_live_env_values(self) -> None:
        example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
        for required in ("QDRANT_URL=", "QDRANT_API_KEY=", "SARVAM_API_KEY=", "DIAGNOSTICS_API_KEY=", "FRONTEND_ORIGINS="):
            self.assertIn(required, example)
        self.assertNotIn("sk-", example)


if __name__ == "__main__":
    unittest.main()
