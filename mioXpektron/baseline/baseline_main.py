from __future__ import annotations
import argparse
import logging
from .baseline_eval import BaselineMethodEvaluator
from .baseline_batch import BaselineBatchCorrector

logger = logging.getLogger(__name__)

# End of module
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ToF‑SIMS baseline evaluation & batch correction")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Evaluate
    pe = sub.add_parser("eval", help="Evaluate methods on spectra files or glob patterns")
    pe.add_argument("files", nargs="+", help="Spectrum files or glob patterns")
    pe.add_argument("--noise-quantile", type=float, default=0.2)
    pe.add_argument("--n-jobs", type=int, default=-1)
    pe.add_argument("--out-dir", default="figures")
    pe.add_argument("--use-preset", action="store_true",
                    help="Use a small parameter preset for common methods.")
    pe.add_argument("--eval-clip-negative", action="store_true",
                    help="If set, allow negatives to be clipped during evaluation (default False).")

    # Run
    pr = sub.add_parser("run", help="Batch baseline correction")
    pr.add_argument("in_dir")
    pr.add_argument("--pattern", default="*.csv")
    pr.add_argument("--recursive", action="store_true")
    pr.add_argument("--method", default="airpls")
    pr.add_argument("--clip-negative", action="store_true")
    pr.add_argument("--n-jobs", type=int, default=-1)
    pr.add_argument("--save-plots", action="store_true")

    args = p.parse_args()

    if args.cmd == "eval":
        ev = BaselineMethodEvaluator(
            files=args.files,
            n_jobs=args.n_jobs,
            eval_clip_negative=args.eval_clip_negative,
            use_small_param_preset=args.use_preset,
        )
        rfzn, nar, snr, bbi, br, nbc, summary = ev.evaluate(noise_quantile=args.noise_quantile)
        logger.info("Best overall method: %s", summary["overall_best_method"])
        ev.plot_publication_figures(out_dir=args.out_dir)
    elif args.cmd == "run":
        runner = BaselineBatchCorrector(args.in_dir, pattern=args.pattern, recursive=args.recursive,
                                       method=args.method, clip_negative=args.clip_negative,
                                       n_jobs=args.n_jobs, save_plots=args.save_plots)
        out_dir = runner.run()
        logger.info("Wrote: %s", out_dir)
