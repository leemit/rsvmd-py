// This work is dedicated to the public domain under the CC0 1.0 Universal license.
// To the extent possible under law, the author has waived all copyright
// and related or neighboring rights to this work.
// https://creativecommons.org/publicdomain/zero/1.0/

use criterion::{criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;

use rsvmd::rsvmd_core::RsvmdProcessor;
use rsvmd::sliding_dft::SlidingDft;
use rsvmd::vmd_core::VmdConfig;

const N: usize = 7200;
const K: usize = 3;

fn make_signal(n: usize) -> Vec<f64> {
    let dt = 1.0 / n as f64;
    (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            (2.0 * PI * 10.0 * t).sin()
                + 0.7 * (2.0 * PI * 50.0 * t).sin()
                + 0.5 * (2.0 * PI * 200.0 * t).sin()
        })
        .collect()
}

fn bench_cold_start(c: &mut Criterion) {
    let signal = make_signal(N);
    let config = VmdConfig {
        alpha: 2000.0,
        k: K,
        tau: 0.0,
        tol: 1e-7,
        window_len: N,
        step_size: 1,
        max_iter: 500,
        ..Default::default()
    };

    c.bench_function("cold_start_N7200_K3", |b| {
        b.iter(|| {
            let mut proc = RsvmdProcessor::new(config.clone());
            let output = proc.initialize(&signal).unwrap();
            std::hint::black_box(output);
        });
    });
}

fn bench_sdft_slide_one(c: &mut Criterion) {
    let signal = make_signal(N + 1);
    let mut sdft = SlidingDft::new(&signal[..N], 0.99999, 0);

    c.bench_function("sdft_slide_one_N7200", |b| {
        b.iter(|| {
            sdft.slide_one(signal[N], signal[0]);
            std::hint::black_box(sdft.spectrum());
        });
    });
}

fn bench_warm_frame(c: &mut Criterion) {
    let signal = make_signal(N + 100);
    let config = VmdConfig {
        alpha: 2000.0,
        k: K,
        tau: 0.0,
        tol: 1e-7,
        window_len: N,
        step_size: 1,
        max_iter: 500,
        ..Default::default()
    };

    let mut proc = RsvmdProcessor::new(config);
    proc.initialize(&signal[..N]).unwrap();
    // Warm up with a few frames
    for i in 0..10 {
        proc.update(&signal[N + i..N + i + 1]).unwrap();
    }

    c.bench_function("warm_frame_N7200_K3", |b| {
        b.iter(|| {
            let output = proc.update(&signal[N + 10..N + 11]).unwrap();
            std::hint::black_box(output);
        });
    });
}

fn bench_e2e_100_frames(c: &mut Criterion) {
    let signal = make_signal(N + 100);
    let config = VmdConfig {
        alpha: 2000.0,
        k: K,
        tau: 0.0,
        tol: 1e-7,
        window_len: N,
        step_size: 1,
        max_iter: 500,
        ..Default::default()
    };

    c.bench_function("e2e_100_frames_N7200_K3", |b| {
        b.iter(|| {
            let mut proc = RsvmdProcessor::new(config.clone());
            proc.initialize(&signal[..N]).unwrap();
            for i in 0..100 {
                proc.update(&signal[N + i..N + i + 1]).unwrap();
            }
            std::hint::black_box(proc.center_freqs());
        });
    });
}

criterion_group!(
    benches,
    bench_cold_start,
    bench_sdft_slide_one,
    bench_warm_frame,
    bench_e2e_100_frames
);
criterion_main!(benches);
