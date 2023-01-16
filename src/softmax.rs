use ndarray::{s, Array1, Array2, Array3};

// NOTE: At the moment also the bias matrix is given
// as input, but it should be initialized with random
// numbers in the future.

fn requantize_row(element: i32, eps_mult: i32, right_shift: i32) -> i8 {
    let shifted = (element * eps_mult) >> right_shift;
    if shifted > 127 {
        return 127;
    } else if shifted < -128 {
        return -128;
    } else {
        return shifted as i8;
    }
}

pub fn requantization_3d(
    m: &mut Array3<i32>,
    m_requant: &mut Array3<i8>,
    eps_mult: i32,
    right_shift: i32,
) {
    println!("===================== 3D Requantization =====================");

    // Loop over the number of heads
    for i in 0..m.shape()[0] {
        // Loop over the head dimension
        for j in 0..m.shape()[1] {
            // print the column of the head matrix
            let row = m.slice(s![i, j, ..]);
            // Iterate over the row and requantize it
            for k in 0..row.len() {
                m_requant[[i, j, k]] = requantize_row(row[k], eps_mult, right_shift);
            }
        }
    }

    // println!("Requantized matrix: {:?}", m_requant);
}

pub fn parallel_requantize3d(
    m: &mut Array3<i32>,
    m_requant: &mut Array2<i8>,
    eps_mult: i32,
    right_shift: i32,
) {
    println!("===================== Parallel 3D Requantization =====================");

    for i in 0..m.shape()[0] {
        for j in 0..m.shape()[1] {
            let row = m.slice(s![i, j, ..]);
            for k in 0..row.len() {
                let shifted = requantize_row(row[k], eps_mult, right_shift) as i32 + m_requant[[i * m.shape()[1] + j, k]] as i32;
                m_requant[[i * m.shape()[1] + j, k]] = requantize_row(shifted, 1, 0);
            }
        }
    }
}

// TODO: Initialize bias matrix with random numbers
pub fn projection_space_transformation(
    p: &mut Array3<i32>,
    m: &mut Array2<i8>,
    w: &mut Array3<i8>,
    b: &mut Array3<i8>,
    bias: u8,
) {
    println!("===================== Projection Space Transformation =====================");
    if bias == 1 {
        for i in 0..p.shape()[0] {
            for j in 0..p.shape()[1] {
                for k in 0..p.shape()[2] {
                    p[[i, j, k]] = b[[i, j, k]] as i32;
                    for l in 0..m.shape()[1] {
                        p[[i, j, k]] += m[[j, l]] as i32 * w[[i, l, k]] as i32;
                    }
                }
            }
        }
    } else {
        for i in 0..p.shape()[0] {
            for j in 0..p.shape()[1] {
                for k in 0..p.shape()[2] {
                    p[[i, j, k]] = 0;
                    for l in 0..m.shape()[1] {
                        p[[i, j, k]] += m[[j, l]] as i32 * w[[i, l, k]] as i32;
                    }
                }
            }
        }
    }

    println!("projected matrix: {:?}", p);
}

pub fn query_key_correlation(
    qp_requant: &mut Array3<i8>,
    kp_requant: &mut Array3<i8>,
    qk: &mut Array3<i32>,
) {
    println!("===================== Query Key Correlation =====================");

    // Loop over the number of heads
    for i in 0..qk.shape()[0] {
        // Loop over the number of queries
        for j in 0..qk.shape()[1] {
            // Loop over the number of keys
            for k in 0..qk.shape()[2] {
                qk[[i, j, k]] = 0;
                // Loop over the number of features
                for l in 0..qk.shape()[1] {
                    qk[[i, j, k]] += qp_requant[[i, j, l as usize]] as i32
                        * kp_requant[[i, k, l as usize]] as i32;
                }
            }
        }
    }

    println!("qk: {:?}", qk);
}

//Compute the approximated softmax function.
pub fn streaming_partial_softmax(
    a_requant: &mut Array3<i8>,
    a_partial_softmax: &mut Array2<i32>,
    seq_len: i32,
) {
    println!("===================== Streaming Partial SoftMax =====================");

    // let log2e: f64 = f64::log2(f64::exp(1.0));
    // let b = 8;
    // let eps_x = b as f64 / (2.0f64.powi(b) * log2e);
    let mut exp_partial_sum = Array1::<i32>::zeros(seq_len as usize);
    let mut max = Array1::<i8>::zeros(64);
    let mut current_max = Array1::<i8>::zeros(64);

    for i in 0..4 {
        let a_requant_slice = a_requant.slice_mut(s![0, .., i * 16..(i + 1) * 16]);

        for n in 0..a_requant_slice.nrows() {
            current_max[[n]] = a_requant_slice.row(n).iter().copied().max().unwrap() as i8;
        }

        for j in 0..seq_len {
            let mut shift_sum;
            if i == 0 || current_max[j as usize] > max[[j as usize]] {
                if i == 0 {
                    shift_sum = 0;
                } else {
                    shift_sum = (current_max[j as usize] - max[[j as usize]]) / 32;
                    if (((current_max[j as usize] - max[[j as usize]]) / 32) - shift_sum) as f64
                        >= 0.5
                    {
                        shift_sum += 1;
                    }
                }
                max[j as usize] = current_max[j as usize];
            } else {
                shift_sum = 0;
            }

            let qb = a_requant
                .slice_mut(s![0, .., i * 16..(i + 1) * 16])
                .mapv(|x| x - max[[j as usize]]);

            let mut qexp = 0;
            for k in 0..qb.ncols() {
                let mut shift = (-qb[[j as usize, k]]) as i32 / 32;
                let shift_int = (-qb[[j as usize, k]]) as i32;

                if shift_int % 32 >= 16 {
                    shift += 1;
                }

                qexp += (2_u32.pow(10) >> shift as i32) as i32;
            }

            exp_partial_sum[[j as usize]] =
                (exp_partial_sum[[j as usize]] >> shift_sum as i32) + qexp;
        }
    }
    for j in 0..seq_len {
        let factor =
            ((2.0f64.powi(8) - 1.0) * 2.0f64.powi(10)) as i32 / exp_partial_sum[j as usize];
        for k in 0..seq_len {
            let mut shift =
                ((max[j as usize] - (a_requant[[0, j as usize, k as usize]])) / 32) as i32;
            let shift_int = max[j as usize] - (a_requant[[0, j as usize, k as usize]]) as i8;
            if shift_int % 32 >= 16 {
                shift += 1;
            }
            a_partial_softmax[[j as usize, k as usize]] =
                (factor as i32) / 2.0f64.powi(shift) as i32;
        }
    }

    println!("a_partial_softmax: {}", a_partial_softmax);
}

pub fn single_head_computation(
    a_partial_softmax: &mut Array2<i32>,
    vp_requant: &mut Array3<i8>,
    o_softmax: &mut Array3<i32>,
) {
    println!("===================== Single Head Computation =====================");

    // Loop over the number of heads
    for i in 0..o_softmax.shape()[0] {
        // Loop over the number of queries
        for j in 0..o_softmax.shape()[1] {
            // Loop over the number of keys
            for k in 0..o_softmax.shape()[2] {
                o_softmax[[i, j, k]] = 0;
                // Loop over the number of features
                for l in 0..o_softmax.shape()[1] {
                    o_softmax[[i, j, k]] +=
                        a_partial_softmax[[j, l]] as i32 * vp_requant[[i, l, k]] as i32;
                }
            }
        }
    }

    println!("o_softmax: {:?}", o_softmax);
}

pub fn multi_head_computation(
    o_softmax_requant: &mut Array3<i8>,
    out: &mut Array3<i32>,
    w_o: &mut Array3<i8>,
    b_o: &mut Array3<i8>,
    bias: u8,
) {
    println!("===================== Multi Head Computation =====================");

    if bias == 1 {
        for i in 0..out.shape()[0] {
            for j in 0..out.shape()[1] {
                for k in 0..out.shape()[2] {
                    out[[i, j, k]] = b_o[[i, j, k]] as i32;
                    for l in 0..out.shape()[1] {
                        out[[i, j, k]] +=
                            o_softmax_requant[[i, j, l]] as i32 * w_o[[i, l, k]] as i32;
                    }
                }
            }
        }
    } else {
        for i in 0..out.shape()[0] {
            for j in 0..out.shape()[1] {
                for k in 0..out.shape()[2] {
                    out[[i, j, k]] = 0;
                    for l in 0..out.shape()[1] {
                        out[[i, j, k]] +=
                            o_softmax_requant[[i, j, l]] as i32 * w_o[[i, l, k]] as i32;
                    }
                }
            }
        }
    }

    println!("out: {:?}", out);
}
