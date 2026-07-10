"""MinerU hybrid ETL — PDF를 우리 청커 입력(content_list.json)으로 뽑는 파이프라인.

noksan_ax의 MinerU 추출 파이프라인을 이식(파이프라인 동일). MinerU 2.5(stock,
`hybrid-http-client` 백엔드)로 raw 산출물을 만들고, pdftext 병합 후처리(결정적)로
content_list.json을 완성한다.

    from llmops_core.etl import run_etl, EtlConfig
    run_etl(Path("doc.pdf"), Path("data"), EtlConfig.from_env())

CLI: `python -m llmops_core.etl doc.pdf -o data --endpoint http://<host>:8002`
"""

from .config import EtlConfig
from .pipeline import run_etl
from .postprocess import postprocess_mineru_output_dir

__all__ = ["EtlConfig", "run_etl", "postprocess_mineru_output_dir"]
