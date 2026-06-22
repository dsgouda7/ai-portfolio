from context_optimizer_benchmark import (
    CompressedIncident,
    build_comparison_metrics,
    run_pipeline_c,
)


def test_run_pipeline_c_returns_branch_selection_and_metrics():
    incident = CompressedIncident(
        core_issue="Checkout timeouts during a deployment window",
        observed_symptoms=[
            "504/499 responses",
            "CosmosDB timeouts",
            "ingress upstream timeouts",
        ],
        technical_identifiers=["CosmosDB", "21012", "ingress-nginx"],
    )

    output, latency, branches, selected_branch, tool_calls, retrieved_lines = run_pipeline_c(
        None,
        incident,
        "mock",
    )

    assert isinstance(output, str) and output
    assert latency >= 0.0
    assert len(branches) == 3
    assert selected_branch in {"cosmos", "ingress", "retry"}
    assert tool_calls >= 0
    assert retrieved_lines >= 0


def test_build_comparison_metrics_returns_experiment_style_metrics():
    metrics = build_comparison_metrics(
        raw_prompt="CosmosDB timeouts in AKS with ingress retry bursts",
        compressed=CompressedIncident(
            core_issue="CosmosDB timeouts during deployment",
            observed_symptoms=["timeouts", "ingress warnings"],
            technical_identifiers=["CosmosDB", "AKS", "21012"],
        ),
        pipe_a_output="CosmosDB timeouts in AKS and ingress retries",
        pipe_c_output="ToT-selected branch: CosmosDB / RU saturation branch.",
        pipe_a_latency=0.12,
        pipe_c_latency=0.08,
        pipe_a_log_lines=2000,
        pipe_c_retrieved_lines=300,
        pipe_c_tool_calls=3,
    )

    assert metrics["pipe_a_prompt_tokens"] > 0
    assert metrics["pipe_c_prompt_tokens"] > 0
    assert metrics["token_reduction_pct"] >= 0.0
    assert 0.0 <= metrics["pipe_a_kw_f1"] <= 1.0
    assert 0.0 <= metrics["pipe_c_judge_score"] <= 1.0
    assert metrics["pipe_c_retrieval_lines"] == 300
    assert metrics["pipe_c_tool_calls"] == 3
