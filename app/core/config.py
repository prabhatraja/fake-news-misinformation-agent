from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    app_name: str = "Fake News Misinformation Detection Agent"
    app_version: str = "0.1.0"
    environment: str = "local"

    model_mode: str = "transformer"
    inference_mode: str = "slm_assisted_hybrid"
    top_k_evidence: int = 5

    runtime_mode: str = "hybrid"

    local_fever_evidence_enabled: bool = True
    local_liar2_evidence_enabled: bool = True

    live_search_enabled: bool = False
    live_search_provider: str = "tavily"
    tavily_api_key: str = ""

    live_search_max_results: int = 5
    live_search_timeout_seconds: int = 10
    live_search_depth: str = "basic"

    live_search_min_relevance: float = 0.55
    live_search_min_trust: float = 0.60
    live_search_min_final_score: float = 0.60

    exa_search_enabled: bool = False
    exa_api_key: str = ""
    exa_search_type: str = "auto"
    exa_max_results: int = 5
    exa_min_final_score: float = 0.58

    # Optional X API configuration. Public oEmbed is used for individual posts.
    # Profile/timeline analysis requires an X API Bearer Token.
    x_bearer_token: str = ""
    x_api_timeout_seconds: int = 15
    x_profile_default_posts: int = 5

    # Local SLM / LLM verifier. This is optional and used only when
    # inference_mode is set to slm_assisted_hybrid or slm_assisted_local.
    slm_enabled: bool = True
    slm_provider: str = "lmstudio"
    slm_base_url: str = "http://127.0.0.1:1234/v1"
    slm_model: str = "qwen/qwen3.5-9b"
    slm_timeout_seconds: int = 180
    slm_temperature: float = 0.0
    slm_max_evidence_items: int = 3
    # Number of top retrieved items semantically judged in the single grounded SLM call.
    # Kept separate from the legacy max-evidence setting so the final verifier can
    # audit all five displayed evidence items without adding extra model calls.
    slm_semantic_evidence_items: int = 5
    slm_max_tokens: int = 1200
    slm_min_confidence_to_override: float = 0.60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AppSettings()


