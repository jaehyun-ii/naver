"""파이프라인 — Argo WorkflowTemplate/ArgoCD Application 선언 + 단계 엔트리포인트.

YAML 선언은 같은 디렉토리(train_eval_deploy.yaml, argocd-app.yaml).
각 DAG 단계는 llmops_core.pipelines.steps.* 의 CLI 엔트리포인트로 실행된다.
"""
