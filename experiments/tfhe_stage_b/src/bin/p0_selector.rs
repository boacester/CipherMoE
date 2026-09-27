use std::env;
use std::io::{self, Write};
use std::time::Instant;

use tfhe::shortint::gen_keys;
use tfhe::shortint::parameters::v1_8::V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
use tfhe::shortint::parameters::{CarryModulus, MessageModulus};

fn public_value(e: usize, k: usize) -> u8 {
    let mut x = (e as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (k as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ 0x7d2a_65f1_93e4_c80b;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    ((x ^ (x >> 31)) % 16) as u8
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    (samples[(samples.len() - 1) / 2] + samples[samples.len() / 2]) / 2.0
}

fn peak_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "usage: p0_selector N repeats [explicit-weight-matrix.json]"
    );
    let n: usize = args[1].parse().unwrap();
    let repeats: usize = args[2].parse().unwrap();
    assert!(n.is_power_of_two() && (2..=64).contains(&n) && repeats > 0);

    let supplied: Option<Vec<Vec<u8>>> = args.get(3).map(|path| {
        let matrix: Vec<Vec<u8>> =
            serde_json::from_str(&std::fs::read_to_string(path).expect("missing matrix file"))
                .expect("expected a JSON array of public vector rows");
        assert_eq!(matrix.len(), n);
        assert!(!matrix[0].is_empty());
        assert!(matrix.iter().all(|row| row.len() == matrix[0].len()));
        assert!(matrix.iter().flatten().all(|&v| v < 16));
        matrix
    });
    let sizes: Vec<usize> = match &supplied {
        Some(matrix) => vec![matrix[0].len()],
        None => vec![1, 4, 8, 16, 32, 64, 128, 256, 512],
    };

    let params = V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
    let capacity = (params.message_modulus.0 * params.carry_modulus.0) as usize;
    let per_rotation = capacity / n;
    let start = Instant::now();
    let (cks, sks) = gen_keys(params);
    let keygen_ms = start.elapsed().as_secs_f64() * 1000.0;
    let key_bytes = bincode::serialized_size(&sks).unwrap();
    let cases: Vec<usize> = if n <= 8 {
        (0..n).collect()
    } else {
        vec![0, n / 2, n - 1]
    };
    let start = Instant::now();
    let encrypted: Vec<_> = cases
        .iter()
        .map(|&e| {
            cks.encrypt_with_message_and_carry_modulus(
                e as u64,
                MessageModulus(n as u64),
                CarryModulus((capacity / n) as u64),
            )
        })
        .collect();
    let encryption_ms = start.elapsed().as_secs_f64() * 1000.0;
    let input_ciphertext_bytes = bincode::serialized_size(&encrypted[0]).unwrap();

    for m in sizes {
        let model: Vec<u8> = match &supplied {
            Some(matrix) => matrix.iter().flatten().copied().collect(),
            None => (0..n)
                .flat_map(|e| (0..m).map(move |k| public_value(e, k)))
                .collect(),
        };
        assert_eq!(model.len(), n * m);
        let functions: Vec<Box<dyn Fn(u64) -> u64 + '_>> = (0..m)
            .map(|k| {
                let table = &model;
                Box::new(move |e: u64| table.get(e as usize * m + k).copied().unwrap_or(0) as u64)
                    as Box<dyn Fn(u64) -> u64>
            })
            .collect();
        let refs: Vec<&dyn Fn(u64) -> u64> = functions.iter().map(|f| f.as_ref()).collect();
        let start = Instant::now();
        let luts: Vec<_> = refs
            .chunks(per_rotation)
            .map(|chunk| sks.generate_many_lookup_table(chunk))
            .collect();
        let mut baseline_luts = Vec::new();
        if m <= 16 {
            baseline_luts = refs.iter().map(|f| sks.generate_lookup_table(f)).collect();
        }
        let preprocessing_ms = start.elapsed().as_secs_f64() * 1000.0;
        let lut_bytes: usize = luts.iter().map(|lut| lut.acc.as_ref().len() * 8).sum();
        let baseline_lut_bytes: usize = baseline_luts
            .iter()
            .map(|lut| lut.acc.as_ref().len() * 8)
            .sum();

        let mut online_ms = Vec::new();
        let mut baseline_ms = Vec::new();
        let mut decrypt_ms = 0.0;
        let mut failures = 0;
        let mut output_ciphertext_bytes = 0;
        for trial in 0..cases.len() + repeats {
            let index = if trial < cases.len() {
                trial
            } else {
                (trial - cases.len()) % cases.len()
            };
            let e = cases[index];
            let ct = &encrypted[index];
            let start = Instant::now();
            let output: Vec<_> = luts
                .iter()
                .flat_map(|lut| sks.apply_many_lookup_table(ct, lut))
                .collect();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(output.len(), m);
            if trial >= cases.len() {
                online_ms.push(elapsed);
            }
            if output_ciphertext_bytes == 0 {
                output_ciphertext_bytes = bincode::serialized_size(&output[0]).unwrap();
            }
            let start = Instant::now();
            for (k, result) in output.iter().enumerate() {
                failures +=
                    usize::from(cks.decrypt_message_and_carry(result) != model[e * m + k] as u64);
            }
            decrypt_ms += start.elapsed().as_secs_f64() * 1000.0;

            if !baseline_luts.is_empty() {
                let start = Instant::now();
                let baseline: Vec<_> = baseline_luts
                    .iter()
                    .map(|lut| sks.apply_lookup_table(ct, lut))
                    .collect();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                if trial >= cases.len() {
                    baseline_ms.push(elapsed);
                }
                let start = Instant::now();
                for (k, result) in baseline.iter().enumerate() {
                    failures += usize::from(
                        cks.decrypt_message_and_carry(result) != model[e * m + k] as u64,
                    );
                }
                decrypt_ms += start.elapsed().as_secs_f64() * 1000.0;
            }
        }
        let online_median_ms = median(online_ms.clone());
        let baseline_median_ms = if baseline_ms.is_empty() {
            None
        } else {
            Some(median(baseline_ms.clone()))
        };
        let row = serde_json::json!({
            "n": n, "m": m, "verified_experts": cases, "verified_outputs": cases.len() * m,
            "failures": failures, "parameter": "V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128",
            "input_capacity": capacity, "outputs_per_blind_rotation": per_rotation,
            "blind_rotations": luts.len(), "keyswitches": luts.len(),
            "sample_extractions": m, "independent_pbs": m,
            "keygen_ms": keygen_ms, "evaluation_key_bytes": key_bytes,
            "encryption_ms": encryption_ms, "preprocessing_ms": preprocessing_ms,
            "online_ms": online_ms, "online_median_ms": online_median_ms,
            "online_ms_per_output": online_median_ms / m as f64,
            "independent_ms": baseline_ms, "independent_median_ms": baseline_median_ms,
            "decrypt_ms": decrypt_ms, "plaintext_model_bytes": model.len(),
            "lut_accumulator_bytes": lut_bytes, "independent_lut_accumulator_bytes": baseline_lut_bytes,
            "input_ciphertext_bytes": input_ciphertext_bytes,
            "output_ciphertext_bytes_each": output_ciphertext_bytes,
            "response_ciphertext_bytes": output_ciphertext_bytes * m as u64,
            "peak_rss_kib": peak_rss_kib(), "custom_weights": supplied.is_some(),
        });
        println!("{row}");
        io::stdout().flush().unwrap();
        assert_eq!(failures, 0, "selector output did not match public vector");
    }
}
