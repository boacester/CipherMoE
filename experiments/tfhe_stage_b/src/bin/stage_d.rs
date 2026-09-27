use std::env;
use std::sync::Arc;
use std::time::Instant;
use tfhe::shortint::ciphertext::{Ciphertext, Degree};
use tfhe::shortint::gen_keys;
use tfhe::shortint::parameters::v1_8::V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;

fn default_weight(e: usize, row: usize, col: usize) -> i64 {
    ((e * 11 + row * 7 + col * 5 + row * col + 1) % 3) as i64 - 1
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
        (5..=6).contains(&args.len()),
        "usage: stage_d d_in d_out tile checks [comma-separated-weights]"
    );
    let d_in: usize = args[1].parse().unwrap();
    let d_out: usize = args[2].parse().unwrap();
    let tile: usize = args[3].parse().unwrap();
    let checks: usize = args[4].parse().unwrap();
    let n = 2_usize;
    assert!(
        d_in > 0 && d_in <= 3 && d_out >= 2 && d_out <= 8 && tile >= 1 && tile <= 8 && checks > 0
    );
    assert!(4 * n * tile <= 64);
    let weights = Arc::new(if let Some(csv) = args.get(5) {
        csv.split(',')
            .map(|s| s.parse::<i64>().expect("invalid public weight"))
            .collect::<Vec<_>>()
    } else {
        (0..n)
            .flat_map(|e| {
                (0..d_out)
                    .flat_map(move |row| (0..d_in).map(move |col| default_weight(e, row, col)))
            })
            .collect::<Vec<_>>()
    });
    assert_eq!(weights.len(), n * d_out * d_in);
    assert!(weights.iter().all(|w| (-1..=1).contains(w)));
    let params = V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
    let start = Instant::now();
    let (cks, sks) = gen_keys(params);
    let keygen_ms = start.elapsed().as_secs_f64() * 1000.0;
    let evaluation_key_bytes = bincode::serialized_size(&sks).unwrap();

    let prep = Instant::now();
    let funcs: Vec<Vec<Box<dyn Fn(u64) -> u64>>> = (0..d_in)
        .map(|col| {
            (0..d_out)
                .map(|row| {
                    let weights = Arc::clone(&weights);
                    Box::new(move |code: u64| {
                        let e = code as usize / 4;
                        if e >= n {
                            return 0;
                        }
                        (weights[(e * d_out + row) * d_in + col] * (code as i64 % 4 - 2))
                            .rem_euclid(16) as u64
                    }) as Box<dyn Fn(u64) -> u64>
                })
                .collect()
        })
        .collect();
    let separate_luts: Vec<Vec<_>> = funcs
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|f| sks.generate_lookup_table(f.as_ref()))
                .collect()
        })
        .collect();
    let tiled_luts: Vec<Vec<_>> = funcs
        .iter()
        .map(|column| {
            column
                .chunks(tile)
                .map(|group| {
                    let refs: Vec<&dyn Fn(u64) -> u64> = group.iter().map(|f| f.as_ref()).collect();
                    sks.generate_many_lookup_table(&refs)
                })
                .collect()
        })
        .collect();
    let preprocessing_ms = prep.elapsed().as_secs_f64() * 1000.0;
    let independent_lut_bytes: usize = separate_luts
        .iter()
        .flatten()
        .map(|lut| lut.acc.as_ref().len() * 8)
        .sum();
    let tiled_lut_bytes: usize = tiled_luts
        .iter()
        .flatten()
        .map(|lut| lut.acc.as_ref().len() * 8)
        .sum();

    let mut failures = 0;
    let mut independent_samples = Vec::new();
    let mut tiled_samples = Vec::new();
    let mut combine_ms = 0.0;
    let mut encrypt_ms = 0.0;
    let mut decrypt_ms = 0.0;
    for trial in 0..checks {
        let e = trial % 2;
        let values: Vec<usize> = (0..d_in)
            .map(|j| {
                let base = trial / (2 * 4_usize.pow(j as u32));
                if d_in > 2 && j == 2 {
                    (base + trial / 2 + j) % 4
                } else {
                    base % 4
                }
            })
            .collect();
        let start = Instant::now();
        let mut encrypted_e = cks.encrypt(e as u64);
        encrypted_e.degree = Degree::new(1);
        let joint: Vec<_> = values
            .iter()
            .map(|&v| {
                let mut x = cks.encrypt(v as u64);
                x.degree = Degree::new(3);
                sks.unchecked_add(&sks.unchecked_scalar_mul(&encrypted_e, 4), &x)
            })
            .collect();
        encrypt_ms += start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let mut baseline: Vec<Vec<Ciphertext>> = Vec::new();
        for col in 0..d_in {
            baseline.push(
                separate_luts[col]
                    .iter()
                    .map(|lut| sks.apply_lookup_table(&joint[col], lut))
                    .collect(),
            );
        }
        let independent_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut packed: Vec<Vec<Ciphertext>> = Vec::new();
        for col in 0..d_in {
            packed.push(
                tiled_luts[col]
                    .iter()
                    .flat_map(|lut| sks.apply_many_lookup_table(&joint[col], lut))
                    .collect(),
            );
        }
        let tiled_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let mut out_baseline = baseline[0].clone();
        let mut out_tiled = packed[0].clone();
        for col in 1..d_in {
            for row in 0..d_out {
                out_baseline[row] = sks.unchecked_add(&out_baseline[row], &baseline[col][row]);
                out_tiled[row] = sks.unchecked_add(&out_tiled[row], &packed[col][row]);
            }
        }
        combine_ms += start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        for row in 0..d_out {
            let expected: i64 = values
                .iter()
                .enumerate()
                .map(|(col, v)| weights[(e * d_out + row) * d_in + col] * (*v as i64 - 2))
                .sum();
            assert!(expected.abs() <= 6);
            let expected_mod = expected.rem_euclid(16) as u64;
            failures +=
                usize::from(cks.decrypt_message_and_carry(&out_baseline[row]) % 16 != expected_mod);
            failures +=
                usize::from(cks.decrypt_message_and_carry(&out_tiled[row]) % 16 != expected_mod);
        }
        decrypt_ms += start.elapsed().as_secs_f64() * 1000.0;
        independent_samples.push(independent_ms);
        tiled_samples.push(tiled_ms);
    }
    let independent_median_ms = median(independent_samples.clone());
    let tiled_median_ms = median(tiled_samples.clone());
    println!("{{\"n\":2,\"d_in\":{d_in},\"d_out\":{d_out},\"tile\":{tile},\"checks\":{checks},\"failures\":{failures},\"keygen_ms\":{keygen_ms},\"evaluation_key_bytes\":{evaluation_key_bytes},\"preprocessing_ms\":{preprocessing_ms},\"independent_lut_bytes\":{independent_lut_bytes},\"tiled_lut_bytes\":{tiled_lut_bytes},\"encrypt_ms\":{encrypt_ms},\"combine_ms\":{combine_ms},\"decrypt_ms\":{decrypt_ms},\"independent_ms\":{independent_samples:?},\"tiled_ms\":{tiled_samples:?},\"independent_median_ms\":{independent_median_ms},\"tiled_median_ms\":{tiled_median_ms},\"independent_pbs\":{},\"tiled_blind_rotations\":{},\"peak_rss_kib\":{}}}", d_in * d_out, d_in * d_out.div_ceil(tile), peak_rss_kib());
    assert_eq!(failures, 0);
}
