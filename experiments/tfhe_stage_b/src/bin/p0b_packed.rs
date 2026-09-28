use std::env;
use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

use tfhe::core_crypto::prelude::{extract_lwe_sample_from_glwe_ciphertext, MonomialDegree};
use tfhe::shortint::gen_keys;
use tfhe::shortint::parameters::v1_8::V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
use tfhe::shortint::parameters::{CarryModulus, Degree, MessageModulus};
use tfhe::shortint::server_key::ManyLookupTableOwned;

const SIZES: [usize; 7] = [1, 8, 32, 64, 128, 256, 512];

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

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

// A selector encrypted in the N-value domain uses N polynomial boxes. Unlike
// generate_many_lookup_table, each box here holds a block of one expert's vector.
fn packed_accumulator(
    template: &ManyLookupTableOwned,
    n: usize,
    model: &[u8],
    m: usize,
    start: usize,
    tile_len: usize,
    stride: usize,
    box_size: usize,
    margin: usize,
) -> ManyLookupTableOwned {
    let mut lut = template.clone();
    lut.sample_extraction_stride = stride;
    lut.per_function_output_degree = vec![Degree::new(15); tile_len];
    let polynomial_size = lut.acc.polynomial_size().0;
    let body = lut.acc.as_mut();
    let body_start = body.len() - polynomial_size;
    let body = &mut body[body_start..];
    body.fill(0);
    let delta = 1u64 << 57; // native u64 torus, 64 plaintext values and one padding bit

    for e in 0..n {
        for k in 0..tile_len {
            let value = model[e * m + start + k] as u64 * delta;
            // Place the value in a disjoint region around its extraction point.
            for j in 0..stride {
                body[e * box_size + margin + k * stride + j] = value;
            }
        }
    }
    lut
}

fn phase_probe(args: &[String]) {
    assert_eq!(args.len(), 4, "usage: p0b_packed probe N trials");
    let n: usize = args[2].parse().unwrap();
    let trials: usize = args[3].parse().unwrap();
    assert_eq!(n, 64, "phase probe currently uses 128-coefficient boxes");
    assert!(trials > 0);
    let params = V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
    let (cks, sks) = gen_keys(params);
    let mut lut = sks.generate_many_lookup_table(&[&|_| 0]);
    let poly_size = lut.acc.polynomial_size().0;
    let body = lut.acc.as_mut();
    let start = body.len() - poly_size;
    for (i, coefficient) in body[start..].iter_mut().enumerate() {
        *coefficient = ((i % 128) / 8) as u64 * (1u64 << 57);
    }
    let experts = if n == 2 {
        vec![0, 1]
    } else {
        vec![1, n / 2, n - 2]
    };
    for e in experts {
        let mut bins = [0usize; 64];
        for _ in 0..trials {
            let mut ct = cks.encrypt_with_message_and_carry_modulus(
                e as u64,
                MessageModulus(n as u64),
                CarryModulus((64 / n) as u64),
            );
            let body = ct.ct.as_mut().last_mut().unwrap();
            *body = body.wrapping_add(64u64 << 50);
            let output = sks.apply_many_lookup_table(&ct, &lut);
            let bin = cks.decrypt_message_and_carry(&output[0]) as usize;
            bins[bin] += 1;
        }
        println!(
            "{}",
            serde_json::json!({"n": n, "e": e, "trials": trials, "phase_center": 64, "bin_width": 8, "decoded_bins": bins.to_vec()})
        );
        io::stdout().flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("probe") {
        phase_probe(&args);
        return;
    }
    assert!(
        (4..=7).contains(&args.len()),
        "usage: p0b_packed N stride repeats [margin] [max_m] [public-vectors.json]"
    );
    let n: usize = args[1].parse().unwrap();
    let stride: usize = args[2].parse().unwrap();
    let repeats: usize = args[3].parse().unwrap();
    assert!([2, 8, 64].contains(&n));
    assert!([1, 2, 4, 8, 16, 32, 64].contains(&stride));
    assert!(repeats > 0);

    let params = V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
    assert_eq!(params.polynomial_size.0, 8192);
    assert_eq!(params.message_modulus.0 * params.carry_modulus.0, 64);
    let box_size = params.polynomial_size.0 / n;
    let margin: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(16);
    let max_m: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(512);
    let supplied: Option<Vec<Vec<u8>>> = args.get(6).map(|path| {
        let matrix: Vec<Vec<u8>> =
            serde_json::from_str(&std::fs::read_to_string(path).expect("missing public vectors"))
                .expect("expected JSON array of unsigned 4-bit vector rows");
        assert_eq!(matrix.len(), n);
        assert!(!matrix[0].is_empty());
        assert!(matrix.iter().all(|row| row.len() == matrix[0].len()));
        assert!(matrix.iter().flatten().all(|&value| value < 16));
        matrix
    });
    assert!(margin * 2 + stride <= box_size);
    let tile_capacity = (box_size - 2 * margin) / stride;
    let phase_offset = margin + stride / 2;
    let start = Instant::now();
    let (cks, sks) = gen_keys(params);
    let keygen_ms = start.elapsed().as_secs_f64() * 1000.0;
    let evaluation_key_bytes = bincode::serialized_size(&sks).unwrap();
    let template = sks.generate_many_lookup_table(&[&|_| 0]);
    let experts: Vec<usize> = if n <= 8 {
        (0..n).collect()
    } else {
        vec![0, 1, n / 4, n / 2, n - 2, n - 1]
    };

    let sizes: Vec<usize> = match &supplied {
        Some(matrix) => vec![matrix[0].len()],
        None => SIZES.into_iter().filter(|&m| m <= max_m).collect(),
    };
    for m in sizes {
        let model: Vec<u8> = match &supplied {
            Some(matrix) => matrix.iter().flatten().copied().collect(),
            None => (0..n)
                .flat_map(|e| (0..m).map(move |k| public_value(e, k)))
                .collect(),
        };
        let start = Instant::now();
        let luts: Vec<_> = (0..m)
            .step_by(tile_capacity)
            .map(|offset| {
                packed_accumulator(
                    &template,
                    n,
                    &model,
                    m,
                    offset,
                    tile_capacity.min(m - offset),
                    stride,
                    box_size,
                    margin,
                )
            })
            .collect();
        let preprocessing_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut times = Vec::new();
        let mut failures = 0;
        let mut tested_outputs = 0;
        let mut output_bytes = 0;
        let mut input_bytes = 0;
        let mut encryption_ms = 0.0;
        let mut decryption_ms = 0.0;
        let mut first_mismatches = Vec::new();

        for trial in 0..(experts.len() + repeats) {
            let e = experts[trial % experts.len()];
            let start = Instant::now();
            let mut input = cks.encrypt_with_message_and_carry_modulus(
                e as u64,
                MessageModulus(n as u64),
                CarryModulus(1),
            );
            encryption_ms += start.elapsed().as_secs_f64() * 1000.0;
            // The shortint encryption API always uses the key's 64-value torus
            // encoding, even if its ciphertext metadata requests a smaller
            // modulus. The client knows e and corrects its phase to N values.
            // The second term is a public offset to center the packed samples.
            let input_delta = 1u64 << (63 - n.trailing_zeros());
            let body = input.ct.as_mut().last_mut().unwrap();
            *body = body
                .wrapping_add((e as u64) * (input_delta - (1u64 << 57)))
                .wrapping_add((phase_offset as u64) << 50);
            if input_bytes == 0 {
                input_bytes = bincode::serialized_size(&input).unwrap();
            }
            let start = Instant::now();
            let mut output: Vec<_> = luts
                .iter()
                .flat_map(|lut| sks.apply_many_lookup_table(&input, lut))
                .collect();
            if trial >= experts.len() {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            assert_eq!(output.len(), m);
            // Input and output have different encodings: the selector occupies N
            // values, but the packed accumulator emits values in a 64-value domain.
            for ct in &mut output {
                ct.message_modulus = params.message_modulus;
                ct.carry_modulus = params.carry_modulus;
            }
            if output_bytes == 0 {
                output_bytes = bincode::serialized_size(&output[0]).unwrap();
            }
            let start = Instant::now();
            for (k, ct) in output.iter().enumerate() {
                let got = cks.decrypt_message_and_carry(ct) as u8;
                let want = model[e * m + k];
                if got != want {
                    failures += 1;
                    if first_mismatches.len() < 6 {
                        first_mismatches
                            .push(serde_json::json!({"e": e, "k": k, "want": want, "got": got}));
                    }
                }
            }
            tested_outputs += m;
            decryption_ms += start.elapsed().as_secs_f64() * 1000.0;
        }

        // Isolate the library's actual sample-extraction primitive on the same
        // GLWE shape. This is a proxy; the online path also clones GLWE/LWE objects.
        let mut sample = cks
            .encrypt_with_message_and_carry_modulus(0, MessageModulus(n as u64), CarryModulus(1))
            .ct;
        let start = Instant::now();
        for lut in &luts {
            for k in 0..lut.function_count() {
                extract_lwe_sample_from_glwe_ciphertext(
                    &lut.acc,
                    &mut sample,
                    MonomialDegree(k * stride),
                );
                black_box(&sample);
            }
        }
        let extraction_proxy_ms = start.elapsed().as_secs_f64() * 1000.0;
        let online_ms = median(times.clone());
        let row = serde_json::json!({
            "n": n, "m": m, "stride": stride, "box_size": box_size,
            "margin": margin, "phase_offset": phase_offset,
            "tile_capacity": tile_capacity, "blind_rotations": luts.len(),
            "blind_rotations_per_output": luts.len() as f64 / m as f64,
            "keyswitches": luts.len(), "sample_extractions": m,
            "online_ms": times, "online_median_ms": online_ms,
            "online_ms_per_output": online_ms / m as f64,
            "extraction_proxy_ms": extraction_proxy_ms,
            "keygen_ms": keygen_ms, "evaluation_key_bytes": evaluation_key_bytes,
            "preprocessing_ms": preprocessing_ms, "encryption_ms": encryption_ms,
            "decryption_ms": decryption_ms, "plaintext_model_bytes": model.len(),
            "lut_accumulator_bytes": luts.iter().map(|lut| lut.acc.as_ref().len() * 8).sum::<usize>(),
            "input_ciphertext_bytes": input_bytes,
            "output_ciphertext_bytes_each": output_bytes,
            "response_ciphertext_bytes": output_bytes * m as u64,
            "verified_experts": experts, "tested_outputs": tested_outputs,
            "failures": failures, "first_mismatches": first_mismatches,
            "custom_weights": supplied.is_some(),
            "parameter": "V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128"
        });
        println!("{row}");
        io::stdout().flush().unwrap();
    }
}
