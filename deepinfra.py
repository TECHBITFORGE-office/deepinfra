from requests import Session
from typing import List, Dict, Generator, Optional
import json


class DeepInfra:
    def __init__(self, api_key: Optional[str] = None):
        self.url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.payload = {}
        self.session = Session()
        self.headers={
            "accept": "text/event-stream",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,hi;q=0.8,ca;q=0.7",
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "content-type": "application/json",
            "origin": "https://deepinfra.com",
            "pragma": "no-cache",
            "referer": "https://deepinfra.com/",
            "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": (
                "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Mobile Safari/537.36"
            ),
            "x-deepinfra-source": "model-embed"
        }
        self.api_key = api_key or None
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # ───── MODEL ALIASES ─────
        self.model_aliases = {

            # ───── Moonshot / Kimi ─────
            "kimi-k2": "moonshotai/Kimi-K2-Instruct",
            "kimi-k2-think": "moonshotai/Kimi-K2-Thinking",
            "kimi-k2-0905": "moonshotai/Kimi-K2-Instruct-0905",

            # ───── MiniMax ─────
            "minimax-m2": "MiniMaxAI/MiniMax-M2",

            # ───── Qwen (General) ─────
            "qwen14": "Qwen/Qwen3-14B",
            "qwen30": "Qwen/Qwen3-30B-A3B",
            "qwen32": "Qwen/Qwen3-32B",
            "qwen235": "Qwen/Qwen3-235B-A22B",
            "qwen235-inst": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "qwen235-think": "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "qwen80-inst": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "qwen80-think": "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "qwq": "Qwen/QwQ-32B",

            # ───── Qwen Coder ─────
            "qwen-coder-30": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "qwen-coder-480": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "qwen-coder-480-turbo": "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",

            # ───── DeepSeek ─────
            "deepseek-r1": "deepseek-ai/DeepSeek-R1",
            "deepseek-r1-turbo": "deepseek-ai/DeepSeek-R1-Turbo",
            "deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
            "deepseek-r1-0528-turbo": "deepseek-ai/DeepSeek-R1-0528-Turbo",
            "deepseek-r1-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-r1-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "deepseek-v3": "deepseek-ai/DeepSeek-V3",
            "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
            "deepseek-v3.1-exp": "deepseek-ai/DeepSeek-V3.2-Exp",
            "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
            "deepseek-v3-0324-turbo": "deepseek-ai/DeepSeek-V3-0324-Turbo",
            "deepseek-terminus": "deepseek-ai/DeepSeek-V3.1-Terminus",
            "deepseek-prover": "deepseek-ai/DeepSeek-Prover-V2-671B",

            # ───── Meta / LLaMA ─────
            "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama11b-vis": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "llama90b-vis": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama70b": "meta-llama/Meta-Llama-3-70B-Instruct",
            "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama3.1-8b-turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
            "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo",
            "llama4-maverick-fp8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",

            # ───── Mistral ─────
            "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
            "mistral-small": "mistralai/Mistral-Small-24B-Instruct-2501",
            "mistral-small-3.1": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "mistral-small-3.2": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "devstral-2505": "mistralai/Devstral-Small-2505",
            "devstral-2507": "mistralai/Devstral-Small-2507",

            # ───── Microsoft ─────
            "phi4": "microsoft/phi-4",
            "phi4-reason": "microsoft/phi-4-reasoning-plus",
            "phi4-mm": "microsoft/Phi-4-multimodal-instruct",
            "wizardlm": "microsoft/WizardLM-2-8x22B",

            # ───── Google ─────
            "gemma4b": "google/gemma-3-4b-it",
            "gemma12b": "google/gemma-3-12b-it",
            "gemma27b": "google/gemma-3-27b-it",
            "gemini-flash": "google/gemini-2.0-flash-001",

            # ───── NVIDIA ─────
            "nemotron70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
            "nemotron-nano": "nvidia/Nemotron-3-Nano-30B-A3B",

            # ───── Zhipu / GLM ─────
            "glm4.5": "zai-org/GLM-4.5",
            "glm4.5-air": "zai-org/GLM-4.5-Air",
            "glm4.5v": "zai-org/GLM-4.5V",
            "glm4.6": "zai-org/GLM-4.6",

            # ───── OpenAI OSS ─────
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "gpt-oss-120b": "openai/gpt-oss-120b",

            # ───── Misc / Research ─────
            "mythomax": "Gryphe/MythoMax-L2-13b",
            "hermes70b": "NousResearch/Hermes-3-Llama-3.1-70B",
            "hermes405b": "NousResearch/Hermes-3-Llama-3.1-405B",
            "skyt1": "NovaSky-AI/Sky-T1-32B-Preview",
            "olmocr": "allenai/olmOCR-7B-0725-FP8",
            "lunaris": "Sao10K/L3-8B-Lunaris-v1-Turbo",
            "euryale70b-v2.2": "Sao10K/L3.1-70B-Euryale-v2.2",
            "euryale70b-v2.3": "Sao10K/L3.3-70B-Euryale-v2.3",
        }

    # ─────────────────────────────────────────────

    def _resolve_model(self, model: str) -> str:
        if model in self.model_aliases:
            return self.model_aliases[model]
        raise ValueError(
            f"Invalid model alias '{model}'. "
            f"Available: {', '.join(self.model_aliases.keys())}"
        )

    # ─────────────────────────────────────────────

    def _payload(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int,
        stream: bool
    ) -> Dict:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True
            }
        }
    # ─────────────────────────────────────────────

    def _stream(self):
        response = self.session.post(
            self.url,
            json=self.payload,
            headers=self.headers,
            stream=True,
            timeout=120
        )
        # print(response.status_code)
        # print(response.content)
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
                except Exception:
                    continue
    # ─────────────────────────────────────────────

    def create(
        self,
        message: List[Dict],
        model: str = "gpt-oss-120b",
        max_tokens: int = 2048,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        resolved_model = self._resolve_model(model)

        self.payload = self._payload(
            model=resolved_model,
            messages=message,
            max_tokens=max_tokens,
            stream=True
        )

        # ───── STREAMING ─────
        if stream:
            return self._stream()
        
        else:
            txt = ""
            for res in self._stream():
                txt += res
            
            return txt 

if __name__ == "__main__":
    client = DeepInfra()

    messages = [
        {"role": "system", "content": "You are a helpful AI."},
        {"role": "user", "content": "Explain transformers in simple words."}
    ]

    for token in client.create(
        messages=messages,
        model="deepseek-r1",
        max_tokens=512,
        stream=True
    ):
        print(token, end="", flush=True)
