use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            let x: f64 = 42.0;
            std::hint::black_box(x);
        });
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
