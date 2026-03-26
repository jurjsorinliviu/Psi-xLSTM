"""
Final revision experiment runner for the IEEE Access acceptance revision.

This script collects the final additional experiments needed for the
camera-ready revision:
1) Re-run the existing reviewer validation suite into a dedicated folder
2) Generate higher-quality publication-ready figures
3) Add cross-device regime analysis for the MOSFET/BJT section
4) Add out-of-expected disturbance sensitivity analysis as a proxy stress test

Default output folder:
    final revision experiments/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_reviewer_experiments import (
    ClusteringStudent,
    DEFAULT_PUBLIC_EXPERIMENTAL_SUITE,
    ReviewerExperimentRunner,
    create_baseline_pinn,
    ensure_dir,
    evaluate_model,
    generate_bjt_dataset,
    generate_memristor_dataset,
    json_safe,
    model_predict,
    set_seed,
    split_dataset,
    train_supervised,
)


def build_reviewer_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    reviewer_args = argparse.Namespace(
        output_dir=str(output_dir),
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quick=args.quick,
        run_all=False,
        sigmoid_vs_exp=False,
        spectral_baselines=False,
        multi_device=False,
        fft_offset=False,
        dc_iv=False,
        active_small_large=False,
        hf_threshold_memristor=args.hf_threshold_memristor,
        hf_threshold_active=args.hf_threshold_active,
        experimental_csv="",
        experimental_input_cols="V",
        experimental_target_col="I",
        experimental_time_col="",
        experimental_group_col="",
        experimental_suite_json="",
        experimental_suite_auto_download=True,
        use_default_public_suite=False,
        auto_install_optional_deps=True,
        write_public_suite_template="",
        use_ngspice_active_data=args.use_ngspice_active_data,
        ngspice_bin=args.ngspice_bin,
        ngspice_mos_model=args.ngspice_mos_model,
        ngspice_timeout_sec=args.ngspice_timeout_sec,
        keep_ngspice_artifacts=args.keep_ngspice_artifacts,
    )
    return reviewer_args


def _styled_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)


class FinalRevisionRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = ensure_dir(Path(args.output_dir))
        self.repo_root = Path(__file__).resolve().parent
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.summary: Dict[str, object] = {
            "config": {
                "device": str(self.device),
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "quick": bool(args.quick),
                "use_ngspice_active_data": bool(args.use_ngspice_active_data),
                "ngspice_mos_model": args.ngspice_mos_model,
            },
            "artifacts": {},
        }

        reviewer_out = self.output_dir / "supporting_validation"
        self.reviewer = ReviewerExperimentRunner(build_reviewer_args(args, reviewer_out))

    def _supporting_results_dir(self) -> Path:
        primary = self.reviewer.output_dir / "exp_sigmoid_vs_exponential" / "sigmoid_vs_exponential_metrics.csv"
        if primary.exists():
            return self.reviewer.output_dir
        fallback = self.repo_root / "reviewer experiments"
        fallback_primary = fallback / "exp_sigmoid_vs_exponential" / "sigmoid_vs_exponential_metrics.csv"
        if fallback_primary.exists():
            return fallback
        raise FileNotFoundError(
            "No supporting validation outputs were found. Run with "
            "--include-supporting-validation or ensure 'reviewer experiments/' exists."
        )

    def run_supporting_validation(self) -> None:
        self.reviewer.run_all()
        self.summary["artifacts"]["supporting_validation"] = str(self.reviewer.output_dir)

    def run_publication_figures(self) -> None:
        out = ensure_dir(self.output_dir / "publication_figures")
        results_dir = self._supporting_results_dir()

        sig_csv = results_dir / "exp_sigmoid_vs_exponential" / "sigmoid_vs_exponential_metrics.csv"
        spec_csv = results_dir / "exp_spectral_baseline_comparison" / "spectral_baseline_comparison.csv"
        multi_dir = results_dir / "exp_multidevice_validation"
        dc_csv = results_dir / "exp_dc_iv_matching" / "dc_iv_rmse.csv"
        small_csv = results_dir / "exp_active_small_large_signal" / "small_signal_bias_frequency_sweep.csv"

        sig = pd.read_csv(sig_csv)
        spec = pd.read_csv(spec_csv).sort_values("hf_mismatch_db")
        dc = pd.read_csv(dc_csv)
        small = pd.read_csv(small_csv)
        mem = pd.read_csv(multi_dir / "memristor_results.csv").sort_values("hf_mismatch_db")
        mos = pd.read_csv(multi_dir / "mosfet_results.csv").sort_values("hf_mismatch_db")
        bjt = pd.read_csv(multi_dir / "bjt_results.csv").sort_values("hf_mismatch_db")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].bar(sig["model"], sig["hf_mismatch_db"], color=["#4C78A8", "#F58518"])
        _styled_axes(axes[0], "Gate Ablation", "Model", "HF mismatch (dB)")
        axes[0].tick_params(axis="x", rotation=12)
        axes[1].bar(sig["model"], sig["rmse"], color=["#4C78A8", "#F58518"])
        _styled_axes(axes[1], "Gate Ablation", "Model", "RMSE")
        axes[1].tick_params(axis="x", rotation=12)
        plt.tight_layout()
        plt.savefig(out / "fig_gate_ablation_summary.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "fig_gate_ablation_summary.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        ax.bar(spec["model"], spec["hf_mismatch_db"], color="#4C78A8")
        _styled_axes(ax, "Spectral-Bias Baseline Comparison", "Model", "HF mismatch (dB)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(out / "fig_spectral_baselines_summary.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "fig_spectral_baselines_summary.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), sharey=True)
        for ax, df, title in zip(axes, [mem, mos, bjt], ["Memristor", "MOSFET", "BJT"]):
            ax.bar(df["model"], df["hf_mismatch_db"], color="#54A24B")
            _styled_axes(ax, title, "Model", "HF mismatch (dB)")
            ax.tick_params(axis="x", rotation=22)
        plt.tight_layout()
        plt.savefig(out / "fig_cross_device_hf_summary.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "fig_cross_device_hf_summary.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
        mos_dc = dc[dc["device"] == "mosfet"]
        bjt_dc = dc[dc["device"] == "bjt"]
        axes[0].plot(mos_dc["curve"], mos_dc["rmse"], marker="o", linewidth=2.5, color="#4C78A8")
        _styled_axes(axes[0], "MOSFET DC Preservation", "Bias condition", "RMSE")
        axes[0].tick_params(axis="x", rotation=25)
        axes[1].plot(bjt_dc["curve"], bjt_dc["rmse"], marker="o", linewidth=2.5, color="#E45756")
        _styled_axes(axes[1], "BJT DC Preservation", "Bias condition", "RMSE")
        axes[1].tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.savefig(out / "fig_dc_iv_summary.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "fig_dc_iv_summary.pdf", bbox_inches="tight")
        plt.close(fig)

        small_mean = small.groupby("freq_hz", as_index=False)[["S21_db_abs_err", "S21_phase_abs_err_deg"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].semilogx(small_mean["freq_hz"], small_mean["S21_db_abs_err"], marker="o", linewidth=2.3)
        _styled_axes(axes[0], "Small-Signal Magnitude Error", "Frequency (Hz)", "|S21| error (dB)")
        axes[1].semilogx(small_mean["freq_hz"], small_mean["S21_phase_abs_err_deg"], marker="o", linewidth=2.3, color="#E45756")
        _styled_axes(axes[1], "Small-Signal Phase Error", "Frequency (Hz)", "Phase error (deg)")
        plt.tight_layout()
        plt.savefig(out / "fig_small_signal_summary.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "fig_small_signal_summary.pdf", bbox_inches="tight")
        plt.close(fig)

        self.summary["artifacts"]["publication_figures"] = [
            "fig_gate_ablation_summary",
            "fig_spectral_baselines_summary",
            "fig_cross_device_hf_summary",
            "fig_dc_iv_summary",
            "fig_small_signal_summary",
        ]
        self.summary["artifacts"]["publication_figures_source"] = str(results_dir)

    def run_cross_device_analysis(self) -> None:
        out = ensure_dir(self.output_dir / "cross_device_analysis")

        mem_ds, mem_dt = self.reviewer._get_memristor_dataset(mode="multitone")
        mos_ds, mos_cfg = self.reviewer._get_mosfet_dataset(mode="multitone")
        bjt_ds, bjt_cfg = self.reviewer._get_bjt_dataset(mode="multitone")

        rows = []
        for name, dataset, dt in [
            ("memristor", mem_ds, mem_dt),
            ("mosfet", mos_ds, mos_cfg.dt),
            ("bjt", bjt_ds, bjt_cfg.dt),
        ]:
            v = dataset["test"]["V"].detach().cpu().numpy()
            i = dataset["test"]["I"].detach().cpu().numpy().reshape(-1)
            t = dataset["test"]["t"].detach().cpu().numpy().reshape(-1)
            v_primary = v[:, 0]

            hysteresis_area = float(np.trapezoid(i, v_primary))
            lag_corr = float(np.corrcoef(i[1:], v_primary[:-1])[0, 1]) if len(i) > 2 else 0.0
            inst_corr = float(np.corrcoef(i, v_primary)[0, 1]) if len(i) > 2 else 0.0
            memory_index = float(abs(lag_corr) - abs(inst_corr))
            linear = np.polyfit(v_primary, i, deg=1)
            i_lin = np.polyval(linear, v_primary)
            nonlinearity_rmse = float(np.sqrt(np.mean((i - i_lin) ** 2)))
            freqs = np.fft.rfftfreq(len(i), d=dt)
            spec = np.abs(np.fft.rfft(i - np.mean(i)))
            spec_energy = spec ** 2
            if np.sum(spec_energy) > 0:
                cum = np.cumsum(spec_energy) / np.sum(spec_energy)
                idx = int(np.searchsorted(cum, 0.9))
                bw90 = float(freqs[min(idx, len(freqs) - 1)])
            else:
                bw90 = 0.0

            rows.append(
                {
                    "device": name,
                    "hysteresis_area": hysteresis_area,
                    "lag_corr": lag_corr,
                    "instant_corr": inst_corr,
                    "memory_index": memory_index,
                    "nonlinearity_rmse": nonlinearity_rmse,
                    "bandwidth_90pct_hz": bw90,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(out / "cross_device_regime_metrics.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].bar(df["device"], df["memory_index"], color=["#72B7B2", "#F58518", "#54A24B"])
        _styled_axes(axes[0], "Temporal-Memory Index", "Device", "abs(lag corr) - abs(inst corr)")
        axes[1].scatter(df["memory_index"], df["nonlinearity_rmse"], s=120, c=["#72B7B2", "#F58518", "#54A24B"])
        for _, row in df.iterrows():
            if row["device"] == "memristor":
                xytext = (-36, 6)
                ha = "right"
            else:
                xytext = (6, 6)
                ha = "left"
            axes[1].annotate(
                row["device"],
                (row["memory_index"], row["nonlinearity_rmse"]),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
            )
        _styled_axes(axes[1], "Regime Map", "Temporal-memory index", "Nonlinearity RMSE")
        plt.tight_layout()
        plt.savefig(out / "cross_device_regime_map.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "cross_device_regime_map.pdf", bbox_inches="tight")
        plt.close(fig)

        self.summary["artifacts"]["cross_device_analysis"] = df.to_dict(orient="records")

    def _build_memristor_models(self, dataset: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.nn.Module]:
        input_dim = dataset["train"]["V"].shape[1] + 1
        return {
            "baseline_pinn": create_baseline_pinn(input_dim=input_dim, hidden_size=96, num_layers=3, output_dim=1),
            "psi_xlstm_clustering": ClusteringStudent(input_dim=input_dim, hidden_size=32, num_layers=2, output_dim=1, num_clusters=3),
        }

    def run_ood_sensitivity_analysis(self) -> None:
        out = ensure_dir(self.output_dir / "ood_sensitivity")
        rng = np.random.RandomState(self.args.seed + 999)

        mem_ds, dt = self.reviewer._get_memristor_dataset(mode="switching_ringing")
        models = self._build_memristor_models(mem_ds)
        trained: Dict[str, torch.nn.Module] = {}
        for name, model in models.items():
            gamma = 0.1 if name == "psi_xlstm_clustering" else 0.0
            train_supervised(
                model,
                mem_ds,
                self.device,
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                lr=self.args.lr,
                structure_gamma=gamma,
            )
            trained[name] = model

        t = mem_ds["test"]["t"].detach().cpu().numpy().reshape(-1)
        v = mem_ds["test"]["V"].detach().cpu().numpy()
        i_nom = mem_ds["test"]["I"].detach().cpu().numpy().reshape(-1)

        amps = [0.02, 0.05, 0.10, 0.20]
        width = max(4, int(0.002 * len(t)))
        center = len(t) // 2

        detailed_rows: List[Dict] = []
        curve_rows: List[Dict] = []
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for idx_amp, amp in enumerate(amps):
            pert = np.zeros_like(i_nom)
            start = max(0, center - width // 2)
            stop = min(len(pert), start + width)
            pert[start:stop] = amp * np.max(np.abs(i_nom) + 1e-9)
            if stop < len(pert):
                tail = np.arange(len(pert) - stop)
                pert[stop:] += 0.3 * amp * np.max(np.abs(i_nom) + 1e-9) * np.exp(-tail / max(width, 1))

            i_ref = i_nom + pert
            for name, model in trained.items():
                with torch.no_grad():
                    pred = model_predict(
                        model,
                        torch.tensor(v, dtype=torch.float32, device=self.device),
                        torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=self.device),
                    ).detach().cpu().numpy().reshape(-1)
                err = np.abs(pred - i_ref)
                peak_err = float(np.max(err))
                post = err[start:]
                threshold = 0.05 * np.max(np.abs(i_ref) + 1e-9)
                below = np.where(post <= threshold)[0]
                recovery_time = float(t[start + below[0]] - t[start]) if len(below) else float(t[-1] - t[start])
                iae = float(np.trapezoid(post, t[start:]))
                detailed_rows.append(
                    {
                        "model": name,
                        "disturbance_amplitude_rel": amp,
                        "peak_abs_error": peak_err,
                        "recovery_time_s": recovery_time,
                        "integrated_abs_error": iae,
                    }
                )

                if amp == amps[len(amps) // 2]:
                    axes[0].plot(t * 1e6, i_ref, color="black", linewidth=1.8, label="Perturbed reference" if name == "baseline_pinn" else None)
                    axes[0].plot(t * 1e6, pred, linewidth=2.0, label=f"{name}")
            curve_rows.append({"t_us": t * 1e6, "reference": i_ref})

        df = pd.DataFrame(detailed_rows)
        df.to_csv(out / "ood_impulse_sensitivity.csv", index=False)
        summary = (
            df.groupby("model", as_index=False)[["peak_abs_error", "recovery_time_s", "integrated_abs_error"]]
            .mean(numeric_only=True)
            .sort_values("integrated_abs_error")
        )
        summary.to_csv(out / "ood_recovery_metrics.csv", index=False)

        marker_map = {"baseline_pinn": "o", "psi_xlstm_clustering": "s"}
        x_offset_map = {"baseline_pinn": -0.003, "psi_xlstm_clustering": 0.003}
        for name, grp in df.groupby("model"):
            grp = grp.sort_values("disturbance_amplitude_rel")
            x = grp["disturbance_amplitude_rel"].to_numpy() + x_offset_map.get(name, 0.0)
            y = grp["recovery_time_s"].to_numpy() + 1e-9
            axes[1].semilogy(
                x,
                y,
                marker=marker_map.get(name, "o"),
                linewidth=2.2,
                markersize=7,
                label=name,
            )

        _styled_axes(axes[0], "Impulse-Like Disturbance Response", "Time (us)", "Current (A)")
        axes[0].legend(fontsize=10)
        _styled_axes(axes[1], "Recovery Time vs Disturbance Amplitude", "Relative disturbance amplitude", "Recovery time (s)")
        axes[1].grid(True, which="both", alpha=0.25)
        axes[1].legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(out / "fig_ood_impulse_response.png", dpi=600, bbox_inches="tight")
        plt.savefig(out / "fig_ood_impulse_response.pdf", bbox_inches="tight")
        plt.close(fig)

        self.summary["artifacts"]["ood_sensitivity"] = {
            "mean_metrics": summary.to_dict(orient="records"),
            "note": "Impulse-like disturbance study used as a proxy for out-of-expected transient upset sensitivity.",
        }

    def write_summary(self) -> None:
        with open(self.output_dir / "final_revision_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(self.summary), f, indent=2)

    def run_all(self) -> None:
        if self.args.include_supporting_validation:
            self.run_supporting_validation()
        self.run_publication_figures()
        self.run_cross_device_analysis()
        self.run_ood_sensitivity_analysis()
        self.write_summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final revision experiments for the IEEE Access acceptance revision.")
    parser.add_argument("--output-dir", type=str, default="final revision experiments")
    parser.add_argument("--device", type=str, default="", help="cpu or cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--publication-figures", action="store_true")
    parser.add_argument("--cross-device-analysis", action="store_true")
    parser.add_argument("--ood-sensitivity", action="store_true")
    parser.add_argument("--include-supporting-validation", action="store_true", help="Re-run the existing reviewer validation suite into a supporting subfolder.")
    parser.add_argument("--use-ngspice-active-data", action="store_true")
    parser.add_argument("--ngspice-bin", type=str, default="C:/ngspice/bin/ngspice.exe")
    parser.add_argument("--ngspice-mos-model", type=str, default="bsim4", choices=["bsim4", "bsim3", "level1"])
    parser.add_argument("--ngspice-timeout-sec", type=int, default=120)
    parser.add_argument("--keep-ngspice-artifacts", action="store_true")
    parser.add_argument("--hf-threshold-memristor", type=float, default=80e3)
    parser.add_argument("--hf-threshold-active", type=float, default=1e6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runner = FinalRevisionRunner(args)

    selected = any([args.publication_figures, args.cross_device_analysis, args.ood_sensitivity, args.include_supporting_validation])
    if args.run_all or not selected:
        runner.run_all()
    else:
        if args.include_supporting_validation:
            runner.run_supporting_validation()
        if args.publication_figures:
            runner.run_publication_figures()
        if args.cross_device_analysis:
            runner.run_cross_device_analysis()
        if args.ood_sensitivity:
            runner.run_ood_sensitivity_analysis()
        runner.write_summary()


if __name__ == "__main__":
    main()
