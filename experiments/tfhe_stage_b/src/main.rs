use std::env;
use std::time::Instant;
use tfhe::shortint::gen_keys;
use tfhe::shortint::parameters::parameters_wopbs_only::{
    LEGACY_WOPBS_ONLY_2_BLOCKS_PARAM_MESSAGE_5_CARRY_0_KS_PBS as P32,
    LEGACY_WOPBS_ONLY_2_BLOCKS_PARAM_MESSAGE_6_CARRY_0_KS_PBS as P64,
    LEGACY_WOPBS_ONLY_2_BLOCKS_PARAM_MESSAGE_7_CARRY_0_KS_PBS as P128,
    LEGACY_WOPBS_ONLY_2_BLOCKS_PARAM_MESSAGE_8_CARRY_0_KS_PBS as P256,
    LEGACY_WOPBS_ONLY_4_BLOCKS_PARAM_MESSAGE_4_CARRY_0_KS_PBS as P16,
};
use tfhe::shortint::wopbs::WopbsKey;

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
        (4..=5).contains(&args.len()),
        "usage: wopbs N checks repeats [comma-separated-weights]"
    );
    let n: usize = args[1].parse().unwrap();
    let checks: usize = args[2].parse().unwrap();
    let repeats: usize = args[3].parse().unwrap();
    assert!(n.is_power_of_two() && (2..=64).contains(&n) && checks > 0 && repeats > 0);
    let weights: Vec<i64> = if let Some(input) = args.get(4) {
        input
            .split(',')
            .map(|value| value.parse().expect("invalid weight"))
            .collect()
    } else {
        (0..n).map(|e| ((e * 7 + 3) % 5) as i64 - 2).collect()
    };
    assert!(weights.len() == n && weights.iter().all(|w| (-2..=2).contains(w)));
    let params = match n {
        2 | 4 => P16,
        8 => P32,
        16 => P64,
        32 => P128,
        64 => P256,
        _ => unreachable!(),
    };
    let p = params.message_modulus.0;
    let setup = Instant::now();
    let (cks, sks) = gen_keys(params);
    let wopbs = WopbsKey::new_wopbs_key_only_for_wopbs(
        cks.as_view().try_into().unwrap(),
        sks.as_view().try_into().unwrap(),
    );
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    // bincode's size traversal does not allocate a serialized key buffer.
    let key_bytes = bincode::serialized_size(&wopbs).unwrap();
    let prep = Instant::now();
    let dummy = cks.encrypt_without_padding(0);
    let lut = wopbs.generate_lut_without_padding(&dummy, |code| {
        let e = (code / 4) as usize;
        if e >= n {
            return 0;
        }
        let x = (code % 4) as i64 - 2;
        (weights[e] * x).rem_euclid(p as i64) as u64
    });
    let lut_bytes = lut.len() * 8;
    let lut: tfhe::shortint::wopbs::ShortintWopbsLUT = lut.into();
    let prep_ms = prep.elapsed().as_secs_f64() * 1000.0;
    let mut verified = 0;
    let mut errors = 0;
    let mut samples = Vec::new();
    let mut encrypt_ms = 0.0;
    let mut decrypt_ms = 0.0;
    for trial in 0..checks + repeats {
        let e = if trial < checks {
            if checks >= n * 4 {
                (trial / 4) % n
            } else {
                trial * (n - 1) / (checks - 1).max(1)
            }
        } else {
            (trial * 17 + 3) % n
        };
        let code = if trial < checks {
            trial % 4
        } else {
            (trial + 1) % 4
        };
        let encrypt = Instant::now();
        let encrypted_e = cks.encrypt_without_padding(e as u64);
        let encrypted_x = cks.encrypt_without_padding(code as u64);
        encrypt_ms += encrypt.elapsed().as_secs_f64() * 1000.0;
        let online = Instant::now();
        let joint = sks.unchecked_add(&sks.unchecked_scalar_mul(&encrypted_e, 4), &encrypted_x);
        let result = wopbs.programmable_bootstrapping_without_padding(&joint, &lut);
        let elapsed = online.elapsed().as_secs_f64() * 1000.0;
        let decrypt = Instant::now();
        let value = cks.decrypt_message_and_carry_without_padding(&result) as i64;
        decrypt_ms += decrypt.elapsed().as_secs_f64() * 1000.0;
        let signed = if value >= p as i64 / 2 {
            value - p as i64
        } else {
            value
        };
        errors += usize::from(signed != weights[e] * (code as i64 - 2));
        verified += 1;
        if trial >= checks {
            samples.push(elapsed);
        }
    }
    samples.sort_by(f64::total_cmp);
    let median = (samples[(repeats - 1) / 2] + samples[repeats / 2]) / 2.0;
    println!("{{\"n\":{n},\"method\":\"tfhe_wopbs_legacy\",\"p\":{p},\"verified\":{verified},\"failures\":{errors},\"keygen_ms\":{setup_ms},\"preprocessing_ms\":{prep_ms},\"encryption_ms\":{encrypt_ms},\"decryption_ms\":{decrypt_ms},\"online_ms\":{samples:?},\"median_ms\":{median},\"evaluation_key_bytes\":{key_bytes},\"lut_bytes\":{lut_bytes},\"peak_rss_kib\":{}}}", peak_rss_kib());
    assert_eq!(errors, 0, "WOPBS did not match integer reference");
}
