"""Method Comparison — head-to-head benchmark of MALMAS, CAAFE, LLM-FE, and Malmus.

Runs all available methods on the same synthetic dataset, evaluates downstream
model performance, and produces:
  1. Side-by-side summary table (features generated, AUC, latency)
  2. Feature overlap analysis via ArtifactDiff
  3. Gain distribution comparison plots
  4. Full HTML dashboard report
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ.setdefault("FF_LOG_LEVEL", "warning")

print("Method Comparison: MALMAS vs CAAFE vs LLM-FE vs Malmus")


def main():
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=5, n_redundant=2,
        n_classes=2, random_state=42,
    )
    feature_names = [f"f{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target"]), df["target"],
        test_size=0.3, random_state=42, stratify=df["target"],
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    baseline_clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
    baseline_clf.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline_clf.predict_proba(X_test)[:, 1])
    print(f"Baseline AUC (no feature engineering): {baseline_auc:.4f}")

    from feature_forge.llm.providers.deepseek import DeepSeekProvider

    llm = DeepSeekProvider(
        model="deepseek-chat",
        api_key=os.environ.get("FF_LLM__API_KEY", ""),
    )

    methods = {}

    from feature_forge.api import FeatureForge
    from feature_forge.config import LLMConfig, Settings

    config = Settings(
        task="classification", metric="auc", n_rounds=1,
        llm=LLMConfig(model="deepseek-chat", api_key=os.environ.get("FF_LLM__API_KEY", "")),
    )
    methods["malmas"] = FeatureForge(config=config, mode="full")

    try:
        from feature_forge.baselines.caafe import CAAFEBaseline
        methods["caafe"] = CAAFEBaseline(llm_client=llm, iterations=2, variant="unified")
    except Exception as exc:
        print(f"CAAFE skipped: {exc}")

    try:
        from feature_forge.baselines.llmfe import LLMFEBaseline
        methods["llmfe"] = LLMFEBaseline(llm_client=llm, n_features=5, mode="single_shot")
    except Exception as exc:
        print(f"LLM-FE skipped: {exc}")

    try:
        from feature_forge.baselines.malmus import MalmusBaseline
        methods["malmus"] = MalmusBaseline(llm_client=llm, n_features=5, mode="single_shot")
    except Exception as exc:
        print(f"Malmus skipped: {exc}")

    print(f"\nMethods to compare: {list(methods.keys())}")

    results = {}
    for name, method in methods.items():
        t0 = time.perf_counter()
        fit_ok = False
        for attempt in range(3):
            try:
                if attempt > 0 and name == "malmas":
                    methods[name] = FeatureForge(config=config, mode="full")
                    method = methods[name]
                method.fit(X_train, y_train)
                fit_ok = True
                break
            except Exception as exc:
                if attempt < 2:
                    print(f"  {name} attempt {attempt+1} failed: {exc}. Retrying...")
                else:
                    latency = round(time.perf_counter() - t0, 2)
                    results[name] = {"status": "error", "latency_s": latency, "error": str(exc)}
                    print(f"  {name}: FAILED after 3 attempts ({exc})")

        if not fit_ok:
            continue

        latency = round(time.perf_counter() - t0, 2)

            X_train_enhanced = method.transform(X_train)
            X_test_enhanced = method.transform(X_test)
            new_cols = [c for c in X_test_enhanced.columns if c not in X_test.columns]

            clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
            clf.fit(X_train_enhanced, y_train)
            enhanced_auc = roc_auc_score(y_test, clf.predict_proba(X_test_enhanced)[:, 1])

            results[name] = {
                "status": "ok",
                "latency_s": latency,
                "n_new_features": len(new_cols),
                "new_feature_names": new_cols[:10],
                "enhanced_auc": enhanced_auc,
                "auc_delta": enhanced_auc - baseline_auc,
                "n_scripts": len(method.generated_scripts),
                "artifacts": method.get_artifacts(),
            }
            print(f"  {name}: {len(new_cols)} features, AUC={enhanced_auc:.4f} ({enhanced_auc - baseline_auc:+.4f}), {latency}s")
        except Exception as exc:
            latency = round(time.perf_counter() - t0, 2)
            results[name] = {"status": "error", "latency_s": latency, "error": str(exc)}
            print(f"  {name}: transform FAILED ({exc})")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    rows = []
    for name, r in results.items():
        if r["status"] == "ok":
            rows.append({
                "Method": name.upper(),
                "Features": r["n_new_features"],
                "Scripts": r["n_scripts"],
                "AUC": f"{r['enhanced_auc']:.4f}",
                "vs Baseline": f"{r['auc_delta']:+.4f}",
                "Latency (s)": r["latency_s"],
            })
        else:
            rows.append({
                "Method": name.upper(),
                "Features": "ERROR",
                "Scripts": "-",
                "AUC": "-",
                "vs Baseline": "-",
                "Latency (s)": r["latency_s"],
            })

    summary_df = pd.DataFrame(rows).set_index("Method")
    print(summary_df.to_string())
    print(f"\nBaseline AUC: {baseline_auc:.4f}")

    from feature_forge.artifacts.schema import ArtifactBundle, FeatureMetadata, ProvenanceRecord
    from feature_forge.artifacts.diff import ArtifactDiff

    bundles = {}
    for name, r in results.items():
        if r["status"] != "ok":
            continue
        arts = r["artifacts"]
        meta_list = []
        prov_records = arts.get("provenance", [])
        for prov in prov_records:
            if isinstance(prov, dict):
                meta_list.append(FeatureMetadata(
                    name=prov.get("feature_name", ""),
                    method=name,
                    agent=prov.get("source_agent"),
                    gain=prov.get("cv_gain"),
                    round=prov.get("round_index"),
                ))

        prov_objs = []
        for p in prov_records:
            if isinstance(p, dict):
                prov_objs.append(ProvenanceRecord(
                    feature_name=p.get("feature_name", ""),
                    source_method=p.get("source_method", name),
                    source_agent=p.get("source_agent"),
                    round_index=p.get("round_index"),
                    cv_gain=p.get("cv_gain"),
                ))

        bundles[name] = ArtifactBundle(
            method_name=name,
            generated_scripts=arts.get("feature_codes", []) or [],
            feature_metadata=meta_list,
            provenance_records=prov_objs,
        )

    if len(bundles) >= 2:
        diff = ArtifactDiff(bundles)
        print("\n--- Artifact Diff Summary ---")
        diff_summary = diff.summary()
        print(f"Total unique features across all methods: {diff_summary['total_unique_features']}")
        print(f"Shared across all methods: {diff_summary['shared_across_all']}")

        for method, data in diff_summary.get("per_method", {}).items():
            print(f"  {method}: {data['total_features']} features, "
                  f"{data['unique_features']} unique, "
                  f"mean_gain={data.get('mean_gain', 'N/A')}")

        overlap = diff.overlap_matrix()
        if not overlap.empty:
            print("\nFeature Overlap Matrix:")
            print(overlap.to_string())

        gains_df = diff.gain_comparison()
        if not gains_df.empty:
            print("\nGain Comparison:")
            print(gains_df.to_string())
    else:
        print("\nNeed >=2 successful methods for diff analysis")
        diff = None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    successful = {n: r for n, r in results.items() if r["status"] == "ok"}
    if successful:
        names = list(successful.keys())
        aucs = [successful[n]["enhanced_auc"] for n in names]
        deltas = [successful[n]["auc_delta"] for n in names]
        feat_counts = [successful[n]["n_new_features"] for n in names]
        latencies = [successful[n]["latency_s"] for n in names]

        ax = axes[0]
        x = range(len(names))
        colors = ["steelblue" if d >= 0 else "salmon" for d in deltas]
        bars = ax.bar(x, aucs, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(y=baseline_auc, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline_auc:.4f})")
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in names], rotation=30)
        ax.set_ylabel("AUC")
        ax.set_title("Downstream AUC by Method")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        ax = axes[1]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        ax.bar(x, feat_counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in names], rotation=30)
        ax.set_ylabel("Number of New Features")
        ax.set_title("Features Generated by Method")

        ax = axes[2]
        ax.bar(x, latencies, color="coral", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in names], rotation=30)
        ax.set_ylabel("Seconds")
        ax.set_title("Fit Latency")

        plt.tight_layout()
        plt.savefig("/tmp/10_method_comparison_overview.png", dpi=120)
        plt.close()
        print("\nOverview plot saved to /tmp/10_method_comparison_overview.png")

    if diff and not diff.gain_comparison().empty:
        gains_df = diff.gain_comparison()
        fig, ax = plt.subplots(figsize=(max(8, len(gains_df) * 0.5), 5))
        gains_df.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Feature Gain by Method")
        ax.set_ylabel("CV Gain")
        ax.set_xlabel("Feature")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Method")
        plt.tight_layout()
        plt.savefig("/tmp/10_gain_comparison.png", dpi=120)
        plt.close()
        print("Gain comparison plot saved to /tmp/10_gain_comparison.png")

    if bundles:
        from feature_forge.artifacts import ArtifactDashboard

        dash = ArtifactDashboard(bundles)
        report_path = "/tmp/10_method_comparison_dashboard.html"
        dash.save(report_path)
        print(f"\nFull dashboard: {report_path} ({os.path.getsize(report_path):,} bytes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
