"""llmops_core — 자체 LLMOps 플랫폼 코어 (design_v2 모듈 조립 방식).

각 서브패키지는 "오픈소스 핵심 모듈 import + 자체 글루 로직"으로 구성된다.
무거운/선택적 의존성(vllm·unsloth·litellm 등)은 각 모듈 내부에서 lazy import 하므로
`import llmops_core`는 코어 의존성만으로 항상 성공한다.
"""

__version__ = "0.1.0"
