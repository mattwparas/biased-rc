use std::{hint::black_box, sync::Arc};

use biased_rc::BiasedRc;

fn main() {
    use std::rc::Rc;
    use std::time::Instant;

    let upper_bound = 10000000;

    let original = Rc::new(10);
    let now = Instant::now();
    for _ in 0..upper_bound {
        black_box({
            let foo = Rc::clone(&original);
            drop(foo);
        });
    }
    println!("Rc clone + drop: {:?}", now.elapsed());

    let original = Arc::new(10);
    let now = Instant::now();
    for _ in 0..upper_bound {
        black_box({
            let foo = Arc::clone(&original);
            drop(foo);
        });
    }
    println!("Arc clone + drop: {:?}", now.elapsed());

    let original = BiasedRc::new(10);
    let now = Instant::now();
    for _ in 0..upper_bound {
        black_box({
            let foo = BiasedRc::clone(&original);
            drop(foo);
        });
    }
    println!("Hybrid clone + drop: {:?}", now.elapsed());
}
