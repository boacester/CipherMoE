use std::collections::HashSet;
use std::env;
use std::time::Instant;
use tfhe::shortint::ciphertext::Degree;
use tfhe::shortint::gen_keys;
use tfhe::shortint::parameters::v1_8::V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;

fn weight(k: usize, e: usize) -> i64 {
    ((e * 7 + k * 3 + (k / 5) * 2 + e * (k / 5) + 3) % 5) as i64 - 2
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

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 4, "usage: stage_c N m repeats");
    let n: usize = args[1].parse().unwrap();
    let m: usize = args[2].parse().unwrap();
    let repeats: usize = args[3].parse().unwrap();
    assert!(n.is_power_of_two() && m.is_power_of_two() && n >= 2 && m > 0 && repeats > 0);
    let params = V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128;
    let p = params.message_modulus.0;
    let domain = 4 * n;
    let capacity = (params.message_modulus.0 * params.carry_modulus.0) as usize;
    assert!(
        domain * m <= capacity,
        "many-LUT domain exceeds accumulator capacity"
    );
    let unique: HashSet<Vec<_>> = (0..m)
        .map(|k| (0..n).map(|e| weight(k, e)).collect())
        .collect();
    assert_eq!(
        unique.len(),
        m,
        "output functions must have distinct public weight rows"
    );

    let started = Instant::now();
    let (cks, sks) = gen_keys(params);
    let keygen_ms = started.elapsed().as_secs_f64() * 1000.0;
    let key_bytes = bincode::serialized_size(&sks).unwrap();
    let prep = Instant::now();
    let funcs: Vec<Box<dyn Fn(u64) -> u64>> = (0..m)
        .map(|k| {
            Box::new(move |code: u64| {
                let e = code as usize / 4;
                if e >= n {
                    return 0;
                }
                (weight(k, e) * (code as i64 % 4 - 2)).rem_euclid(16) as u64
            }) as Box<dyn Fn(u64) -> u64>
        })
        .collect();
    let refs: Vec<&dyn Fn(u64) -> u64> = funcs.iter().map(|f| f.as_ref()).collect();
    let singles: Vec<_> = refs.iter().map(|f| sks.generate_lookup_table(f)).collect();
    let many = sks.generate_many_lookup_table(&refs);
    let preprocessing_ms = prep.elapsed().as_secs_f64() * 1000.0;
    let single_lut_bytes = singles
        .iter()
        .map(|x| (x.acc.as_ref().len() * 8) as u64)
        .sum::<u64>();
    let many_lut_bytes = (many.acc.as_ref().len() * 8) as u64;

    let mut independent_ms = Vec::new();
    let mut many_ms = Vec::new();
    let mut failures = 0;
    let mut encryption_ms = 0.0;
    let mut decryption_ms = 0.0;
    // Exhaustive for the configured N; additional timed trials cover both invocation orders.
    for trial in 0..domain + repeats {
        let e = (trial / 4) % n;
        let code = trial % 4;
        let enc = Instant::now();
        let mut encrypted_e = cks.encrypt(e as u64);
        let mut encrypted_x = cks.encrypt(code as u64);
        // Public bounds from the client input contract; independent of secret values.
        encrypted_e.degree = Degree::new((n - 1) as u64);
        encrypted_x.degree = Degree::new(3);
        encryption_ms += enc.elapsed().as_secs_f64() * 1000.0;
        let joint = sks.unchecked_add(&sks.unchecked_scalar_mul(&encrypted_e, 4), &encrypted_x);
        let (separate, packed, separate_elapsed, packed_elapsed) = if trial % 2 == 0 {
            let start = Instant::now();
            let separate: Vec<_> = singles
                .iter()
                .map(|lut| sks.apply_lookup_table(&joint, lut))
                .collect();
            let a = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let packed = sks.apply_many_lookup_table(&joint, &many);
            let b = start.elapsed().as_secs_f64() * 1000.0;
            (separate, packed, a, b)
        } else {
            let start = Instant::now();
            let packed = sks.apply_many_lookup_table(&joint, &many);
            let b = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let separate: Vec<_> = singles
                .iter()
                .map(|lut| sks.apply_lookup_table(&joint, lut))
                .collect();
            let a = start.elapsed().as_secs_f64() * 1000.0;
            (separate, packed, a, b)
        };
        let dec = Instant::now();
        for k in 0..m {
            let expected = (weight(k, e) * (code as i64 - 2)).rem_euclid(16) as u64;
            failures += usize::from(cks.decrypt_message_and_carry(&separate[k]) % 16 != expected);
            failures += usize::from(cks.decrypt_message_and_carry(&packed[k]) % 16 != expected);
        }
        decryption_ms += dec.elapsed().as_secs_f64() * 1000.0;
        if trial >= domain {
            independent_ms.push(separate_elapsed);
            many_ms.push(packed_elapsed);
        }
    }
    let independent_median = median(independent_ms.clone());
    let many_median = median(many_ms.clone());
    println!("{{\"n\":{n},\"m\":{m},\"p\":{p},\"capacity\":{capacity},\"verified_inputs\":{domain},\"verified_outputs\":{},\"failures\":{failures},\"keygen_ms\":{keygen_ms},\"evaluation_key_bytes\":{key_bytes},\"preprocessing_ms\":{preprocessing_ms},\"single_lut_bytes\":{single_lut_bytes},\"many_lut_bytes\":{many_lut_bytes},\"encryption_ms\":{encryption_ms},\"decryption_ms\":{decryption_ms},\"independent_ms\":{independent_ms:?},\"many_ms\":{many_ms:?},\"independent_median_ms\":{independent_median},\"many_median_ms\":{many_median},\"independent_pbs\":{m},\"many_blind_rotations\":1,\"peak_rss_kib\":{}}}", domain * m * 2, peak_rss_kib());
    assert_eq!(failures, 0);
}
