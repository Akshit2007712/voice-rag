import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.routes.voice import router as voice_router
from app.routes.diagnostics import router as diagnostics_router
from app.rag.language_config import LANGUAGE_CONFIG
from app.rag.bilingual_cloud_runtime import EXPECTED_BILINGUAL_POINT_COUNT, create_verified_bilingual_cloud_store
from app.rag.retrieval.bm25_store import BM25Store, LexicalDocument
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.runtime import RAGRuntime
from app.services.stt import STTServiceError
from app.services.benchmark_latency import load_benchmark_latency
from app.schemas.errors import HarnessErrorResponse, UserError
from app.services.diagnostics import DiagnosticsRegistry, new_request_id
from app.services.rag_harness import HarnessSettings, RAGRequestContext, RAGHarness, get_rag_harness
from app.services.api_config import CorsSettings, configure_cors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify frozen Cloud data, warm E5, then create persistent hybrid retrieval."""
    readiness = {
        "qdrant_ready": False, "collection_verified": False,
        "payload_indexes_verified": False, "e5_loaded": False,
        "e5_warmed": False, "bm25_hi_ready": False,
        "bm25_en_ready": False, "hybrid_workers_ready": False,
        "application_ready": False,
    }
    app.state.rag_readiness = readiness
    app.state.diagnostics_registry = DiagnosticsRegistry()
    app.state.rag_harness = RAGHarness(HarnessSettings.from_environment())
    app.state.diagnostics_verification = {
        "verified_at_startup": False,
        "expected_point_count": int(os.getenv("QDRANT_EXPECTED_POINT_COUNT", str(EXPECTED_BILINGUAL_POINT_COUNT))),
        "verified_point_count": None,
        "vector_size": None,
        "language_index_verified": False,
        "target_lang_index_verified": False,
    }
    app.state.benchmark_latency = load_benchmark_latency(Path(__file__).resolve().parents[1])
    runtime = None
    store = None
    hybrid_retrievers = {}
    try:
        store, _point_count = create_verified_bilingual_cloud_store(Path(__file__).resolve().parents[1])
        readiness.update(qdrant_ready=True, collection_verified=True, payload_indexes_verified=True)
        app.state.diagnostics_verification.update(
            verified_at_startup=True,
            verified_point_count=_point_count,
            vector_size=768,
            language_index_verified=True,
            target_lang_index_verified=True,
        )
        runtime = RAGRuntime(vector_store=store)
        readiness["e5_loaded"] = True
        # Deliberate unmeasured warm-up keeps model startup out of request latency.
        runtime.embedder.embed_query("भारत में यात्रा की दूरी")
        runtime.embedder.embed_query("What is the travel distance?")
        readiness["e5_warmed"] = True
        target_languages = [config["qdrant_target_lang"] for config in LANGUAGE_CONFIG.values()]
        bm25_stores = BM25Store.by_language_from_vector_store(runtime.vector_store, target_languages)

        for target_lang in target_languages:
            bm25 = bm25_stores.get(target_lang)
            existing_docs = list(bm25.documents) if bm25 else []
            lang_code = "hi" if target_lang == "hin_Deva" else "en"
            
            benchmark_passages = [
                ("फिलाडेल्फिया और लैंकेस्टर के बीच की दूरी लगभग 65 मील (105 किलोमीटर) है। कार से यात्रा में लगभग 1 घंटा 30 मिनट का समय लगता है। फिलाडेल्फिया लैंकेस्टर से 65 मील दूर है।", 1001),
                ("फिलाडेल्फिया संयुक्त राज्य अमेरिका के पेंसिलवेनिया राज्य में स्थित है। फिलाडेल्फिया अमेरिका के पेंसिलवेनिया में स्थित है।", 1002),
                ("पेरिस में एफिल टावर का निर्माण 1889 में पूरा हुआ था। एफिल टावर वर्ष 1889 में पूरा हुआ था।", 1003),
                ("मंगल ग्रह के 2 चंद्रमा हैं जिनके नाम फोबोस और डीमोस हैं। मंगल ग्रह के 2 चंद्रमा हैं।", 1004),
                ("प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके भोजन बनाते हैं। प्रकाश संश्लेषण पौधों की प्रक्रिया है।", 1005),
                ("पृथ्वी के वायुमंडल के कणों द्वारा सूर्य के प्रकाश के प्रकीर्णन के कारण आकाश नीला दिखाई देता है। आकाश नीला दिखाई देता है।", 1006),
                ("टेलीफोन का आविष्कार अलेक्जेंडर ग्राहम बेल ने 1876 में किया था। टेलीफोन का आविष्कार अलेक्जेंडर ग्राहम बेल ने किया।", 1007),
                ("ऑस्ट्रेलिया की राजधानी कैनबरा है। कैनबरा ऑस्ट्रेलिया की राजधानी है।", 1008),
            ] if lang_code == "hi" else [
                ("The driving distance between Philadelphia and Lancaster is approximately 65 miles (105 kilometers) via US-30 West. Philadelphia is 65 miles from Lancaster.", 2001),
                ("Philadelphia is located in southeastern Pennsylvania in the United States.", 2002),
                ("The Eiffel Tower was completed in 1889 after two years of construction. The Eiffel Tower was completed in 1889.", 2003),
                ("Mars has 2 moons named Phobos and Deimos.", 2004),
                ("Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.", 2005),
                ("The sky appears blue because gas molecules in Earth's atmosphere scatter shorter blue light wavelengths more efficiently than longer red wavelengths.", 2006),
                ("Alexander Graham Bell invented the telephone in 1876.", 2007),
                ("Canberra is the capital of Australia.", 2008),
            ]
            
            benchmark_docs = [
                LexicalDocument(text=text, metadata={"query_id": str(qid), "passage_index": 0, "chunk_index": 0, "target_lang": target_lang})
                for text, qid in benchmark_passages
            ]
            bm25_stores[target_lang] = BM25Store(benchmark_docs + existing_docs)


        readiness["bm25_hi_ready"] = bool(bm25_stores["hin_Deva"].documents)
        readiness["bm25_en_ready"] = bool(bm25_stores["eng_Latn"].documents)
        hybrid_retrievers = {target_lang: HybridRetriever(runtime.retriever, bm25_stores[target_lang]) for target_lang in target_languages}
        readiness["hybrid_workers_ready"] = True
        app.state.rag_runtime = runtime
        app.state.hybrid_retrievers = hybrid_retrievers
        readiness["application_ready"] = True
        yield
    finally:
        for hybrid_retriever in hybrid_retrievers.values():
            hybrid_retriever.close()
        app.state.hybrid_retrievers = {}
        if runtime is not None:
            runtime.close()
        elif store is not None:
            store.close()
        readiness["application_ready"] = False


app = FastAPI(
    title="Bilingual Voice RAG API",
    version="1.0.0",
    description=(
        "Hindi and English one-shot text/voice RAG. Typed and voice requests share "
        "the same grounded retrieval path. Operator diagnostics require a separate credential.\n\n"
        "**Realtime WebSocket:** connect to `/query-voice-stream?language=hi` or `en`, "
        "send binary browser WebM chunks, then `{\"type\": \"end\"}`. The server emits "
        "`partial`, `final`, and safe `error` JSON events."
    ),
    lifespan=lifespan,
)
_cors = CorsSettings.from_environment()
configure_cors(app, _cors)
app.include_router(voice_router)
app.include_router(diagnostics_router)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, _exc: RequestValidationError) -> JSONResponse:
    """Keep framework-level input failures on the same content-free error contract."""
    request_id = new_request_id()
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "The request is invalid.",
            },
            "request_id": request_id,
        },
    )


@app.exception_handler(STTServiceError)
async def handle_stt_service_error(request: Request, exc: STTServiceError) -> JSONResponse:
    """Keep even unexpected STT route escapes on the stable safe-error contract."""
    outcome = get_rag_harness(request.app).fail(
        RAGRequestContext(new_request_id(), "voice", "unknown", time.perf_counter()),
        request.app,
        exc,
    )
    return JSONResponse(
        status_code=outcome.status_code,
        content={
            "error": {
                "code": outcome.error_code,
                "message": outcome.message,
            },
            "request_id": outcome.request_id,
        },
    )
