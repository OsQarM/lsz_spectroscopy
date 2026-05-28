import numpy as np
import matplotlib.pyplot as plt

from fourier import fourier_analysis, fit_decay_rates
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
from turnpike import solve_incomplete_turnpike


def run_full_diagnostics(pc_list, tw_l, nqubits, epsilon, H_target_dict,
                         ramp_time, dt, egvals_midpoint, w_noise=False,
                         t1_rates=None, t2_rates=None):
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
    experimental_freqs, experimental_phases = fourier_analysis(
        pc_list, tw_l,
        n_peaks=(2**nqubits) * (2**nqubits - 1) // 2,
        prominence_threshold=0.05, zero_pad_factor=32, window='hann'
    )

    # 2. Global fit
    experimental_amps, lambdas, dc_fit = fit_decay_rates(
        pc_list, tw_l, experimental_freqs, experimental_phases, noise=w_noise
    )

    print("Amplitudes:", experimental_amps)
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
    scatter_plot(matched_detected, matched_true, residuals)

    results.update({
        'energy_diffs': experimental_energy_diffs,
        'true_differences': true_differences,
        'matched_detected': matched_detected,
        'matched_true': matched_true,
    })

    # 4. Turnpike spectrum reconstruction
    solutions = solve_incomplete_turnpike(experimental_energy_diffs, n_levels=2**nqubits)
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
    validation_exp = LSZ_experiment(nqubits, epsilon, H_target_dict, ramp_time, 0, dt)
    target_H = validation_exp.H_numpy(validation_exp.tr)
    u_exact_amplitudes, u_exact_phases_rel, egvals_t, egvecs_t = validate_state_vector(
        validation_exp, target_H
    )

    print("Exact |u_m|:", u_exact_amplitudes)
    print("Exact phases (mod pi, rel to u_0):", u_exact_phases_rel)

    # u_m scatter
    plt.figure()
    plt.scatter(np.sort(u_m), np.sort(u_exact_amplitudes))
    diag = np.linspace(np.min(u_m), np.max(u_m), 100)
    plt.plot(diag, diag, "--", c="red", label="true = estimate")
    plt.xlabel("estimated u_m")
    plt.ylabel("true u_m")
    plt.legend(frameon=False)
    plt.show()

    # phi_m scatter
    plt.figure()
    plt.scatter(np.sort(phi_m), np.sort(u_exact_phases_rel))
    diag = np.linspace(np.min(phi_m), np.max(phi_m), 100)
    plt.plot(diag, diag, "--", c="red", label="true = estimate")
    plt.xlabel("estimated phi_m")
    plt.ylabel("true phi_m")
    plt.legend(frameon=False)
    plt.show()

    results.update({
        'u_exact_amplitudes': u_exact_amplitudes,
        'u_exact_phases_rel': u_exact_phases_rel,
        'egvecs_target': egvecs_t,
    })

    # 7. Dephasing rate comparison (optional)
    if t2_rates is not None:
        possible_lambdas = calculate_dephasing_rates_upper_bound(t2_rates)
        exact_lambdas = predict_gamma_rates(t1_rates, t2_rates)
        plot_dephasing_comparison(lambdas, exact_lambdas)
        print("Exact lambdas (predicted):", exact_lambdas)
        print("Experimental lambdas:", lambdas)

        results.update({
            'possible_lambdas': possible_lambdas,
            'exact_lambdas': exact_lambdas,
        })

    return results
