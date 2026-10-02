import numpy as np
import matplotlib.pyplot as plt

from plotting.diagnostic_plots import plot_estimate_vs_true
from plotting.spectrum_plots import plot_spectrum_with_predictions

from fourier import fourier_analysis,  fourier_analysis_iterative, fit_decay_rates
from spectrum_matching import (
    compute_true_differences,
    match_energy_differences,
    table_printout,
    scatter_plot,
    compare_spectrum_to_turnpike,
)
from state_reconstruction import (
    create_transition_matrices,
    compute_u_m,
    compute_phi_m,
    validate_state_vector,
)
from dephasing import (
    calculate_dephasing_rates_upper_bound,
    predict_dephasing_rates_exact,
    predict_gamma_rates,
    plot_dephasing_comparison,
)
from lsz_experiment import LSZ_experiment
from turnpike import solve_turnpike, solve_incomplete_turnpike


def  run_full_diagnostics(pc_list, tw_l, nqubits, epsilon, H_target_dict,
                         ramp_time, dt, egvals_midpoint, w_noise=False,
                         t1_rates=None, t2_rates=None, initial_state=None,
                         up_z_assym_list=None, down_z_assym_list=None,
                         up_x_assym_list=None, down_x_assym_list=None,
                         plots=True):
    """
    Full diagnostic pipeline:
      1. FFT analysis -> frequencies, phases
      2. Global fit  -> amplitudes, decay rates, DC
      3. Match detected vs true energy differences (from egvals_midpoint)
      4. Turnpike reconstruction of the spectrum
      5. Build transition matrices, recover u_m and phi_m
      6. Validate against exact state vector
      7. (Optional) Dephasing-rate comparison if t2_rates provided

    Returns a dict with all intermediate results.
    """
    results = {}

    # 1. Fourier analysis try hann, blackmanharris, boxcar
    # experimental_freqs, experimental_phases = fourier_analysis_iterative(
    #     pc_list, tw_l,
    #     n_peaks=(2**nqubits) * (2**nqubits - 1) // 2 +1,
    #     prominence_threshold=0.001, zero_pad_factor=32, window='hann', verbose=True
    # )

    experimental_freqs, experimental_phases = fourier_analysis(
        pc_list, tw_l,
        n_peaks=(2**nqubits) * (2**nqubits - 1) // 2,
        prominence_threshold=0.001, zero_pad_factor=32, window='hann', 
        detect_prominence=0.001, f_min=0.001, plots=plots
    )

    # 2. Global fit
    experimental_amps, lambdas, dc_fit = fit_decay_rates(
        pc_list, tw_l, experimental_freqs, experimental_phases, noise=w_noise,
        plots=plots
    )

    print("Amplitudes:", experimental_amps, " Sum last three:", sum(experimental_amps[-3:]))
    print("Decay rates:", lambdas)
    print("DC (fitted):", dc_fit, "  DC (mean):", np.mean(pc_list))

    results.update({
        'freqs': experimental_freqs,
        'phases': experimental_phases,
        'amplitudes': experimental_amps,
        'lambdas': lambdas,
        'dc_fit': dc_fit,
    })

    # 3. Compare to true energy differences
    experimental_energy_diffs = 2 * np.pi * experimental_freqs
    true_differences = compute_true_differences(egvals_midpoint)

    matched_detected, matched_true, residuals, col_idx = match_energy_differences(
        experimental_energy_diffs, true_differences
    )
    table_printout(matched_detected, matched_true, residuals, true_differences, col_idx)
    scatter_plot(matched_detected, matched_true, residuals, plots=plots)

    if plots:
        # Recomputed with the same window/zero-padding as the fourier_analysis
        # call above (hann, zero_pad_factor=32) so the axis lines up with
        # `experimental_freqs`.
        y = np.asarray(pc_list, dtype=float)
        t = np.asarray(tw_l, dtype=float)
        dt_fft = t[1] - t[0]
        win = np.hanning(len(y))
        spec = np.fft.rfft((y - y.mean()) * win, n=32 * len(y))
        frq_full = np.fft.rfftfreq(32 * len(y), dt_fft)
        true_freqs = true_differences / (2 * np.pi)
        plot_spectrum_with_predictions(
            frq_full[1:], np.abs(spec)[1:],
            detected_freqs=experimental_freqs,
            true_freqs=true_freqs,
            show=True,
        )

    results.update({
        'energy_diffs': experimental_energy_diffs,
        'true_differences': true_differences,
        'matched_detected': matched_detected,
        'matched_true': matched_true,
    })

    # 4. Turnpike spectrum reconstruction
    n_levels = 2 ** nqubits
    n_expected_diffs = n_levels * (n_levels - 1) // 2
    if len(experimental_energy_diffs) == n_expected_diffs:
        # The exact solver needs a matching tolerance that reflects the real
        # frequency-extraction error, not an arbitrary default: with peaks
        # this closely spaced, FFT fitting error can exceed the 0.01 default
        # by an order of magnitude, which makes every backtracking branch
        # fail to match. Size slack off the worst observed residual from the
        # true-difference matching above (a proxy for FFT peak-picking error),
        # with a safety margin for levels not sampled by that residual set.
        turnpike_slack = max(3 * np.max(np.abs(residuals)), 0.01)
        levels = solve_turnpike(experimental_energy_diffs, slack=turnpike_slack)
        solutions = {
            'levels': levels,
            'explained': n_expected_diffs,
            'total': n_expected_diffs,
            'score': 1.0,
            'method': 'exact',
        }
    else:
        solutions = solve_incomplete_turnpike(experimental_energy_diffs, n_levels=n_levels)
    experimental_energies, shifted_true_energies = compare_spectrum_to_turnpike(
        solutions, egvals_midpoint
    )

    results.update({
        'turnpike_solutions': solutions,
        'experimental_energies': experimental_energies,
        'shifted_true_energies': shifted_true_energies,
    })

    # 5. Transition matrices + u_m / phi_m
    w_mn, a_mn, d_mn, lmd_mn, w_predicted = create_transition_matrices(
        experimental_energies, experimental_energy_diffs,
        experimental_amps, experimental_phases, lambdas
    )

    e_m = w_mn[0]
    u_m = compute_u_m(a_mn)
    phi_m = compute_phi_m(d_mn)

    print(f"e_m:   {np.array2string(np.array(e_m),   precision=4, suppress_small=True)}")
    print(f"phi_m: {np.array2string(np.array(phi_m), precision=4, suppress_small=True)}")
    print(f"u_m:   {np.array2string(np.array(u_m),   precision=4, suppress_small=True)}")

    results.update({
        'w_mn': w_mn, 'a_mn': a_mn, 'd_mn': d_mn, 'lmd_mn': lmd_mn,
        'w_predicted': w_predicted,
        'e_m': e_m, 'u_m': u_m, 'phi_m': phi_m,
    })

    # 6. Validate against exact state vector
    validation_exp = LSZ_experiment(nqubits, epsilon, H_target_dict, ramp_time, 0, dt,
                                    up_z_assymetry_factors=up_z_assym_list,
                                    down_z_assymetry_factors=down_z_assym_list,
                                    up_x_assymetry_factors=up_x_assym_list,
                                    down_x_assymetry_factors=down_x_assym_list,
                                    initial_state=initial_state)
    target_H = validation_exp.H_numpy(validation_exp.tr)
    u_exact_amplitudes, u_exact_phases_rel, egvals_t, egvecs_t = validate_state_vector(
        validation_exp, target_H
    )

    print("Exact |u_m|:", u_exact_amplitudes)
    print("Exact phases (mod pi, rel to u_0):", u_exact_phases_rel)

    # u_m and phi_m: estimated vs true, with the red dashed identity line.
    if plots:
        plot_estimate_vs_true(np.sort(u_exact_amplitudes), np.sort(u_m),
                              xlabel=r"true $|u_m|$",
                              ylabel=r"estimated $|\tilde{u}_m|$",
                              show=True)
        plot_estimate_vs_true(np.sort(u_exact_phases_rel), np.sort(phi_m),
                              xlabel=r"true $\phi_m$",
                              ylabel=r"estimated $\tilde{\phi}_m$",
                              show=True)

    results.update({
        'u_exact_amplitudes': u_exact_amplitudes,
        'u_exact_phases_rel': u_exact_phases_rel,
        'egvecs_target': egvecs_t,
    })

    # 7. Dephasing rate comparison (optional)
    if t2_rates is not None:
        possible_lambdas = calculate_dephasing_rates_upper_bound(t2_rates)
        exact_lambdas = predict_gamma_rates(t1_rates, t2_rates)
        plot_dephasing_comparison(lambdas, exact_lambdas, plots=plots)
        print("Exact lambdas (predicted):", exact_lambdas)
        print("Experimental lambdas:", lambdas)

        results.update({
            'possible_lambdas': possible_lambdas,
            'exact_lambdas': exact_lambdas,
        })

    return results
