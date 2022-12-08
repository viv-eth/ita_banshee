use ndarray:: {
    prelude::*,
    s,
    Array,
    Array1,
    Array2,
    Array3,
    linalg::general_mat_mul,
};

// NOTE: At the moment also the bias matrix is given 
// as input, but it should be initialized with random
// numbers in the future. 

pub fn requantization_3d(M: &mut Array3<i32>, M_requant: &mut Array3<i8>, eps_mult: i32, right_shift: i32) {

    println!("===================== 3D Requantization =====================");

    // Loop over the number of heads
    for i in 0..M.shape()[0] {
        // Loop over the head dimension
        for j in 0..M.shape()[1] {
            // print the column of the head matrix
            let row = M.slice(s![i, j, ..]);
            // Iterate over the row and requantize it
            for k in 0..row.len() {
                let shifted = (row[k] * eps_mult) >> right_shift;
                if shifted > 127 {
                    M_requant[[i, j, k]] = 127;
                } else if shifted < -128 {
                    M_requant[[i, j, k]] = -128;
                } else {
                    M_requant[[i, j, k]] = shifted as i8;
                }
            }
        }
        
    }

    // println!("Requantized matrix: {:?}", M_requant);
}

// TODO: Initialize bias matrix with random numbers
pub fn query_projection_space_transformation(Q_p: &mut Array3<i32>, Q: &mut Array2<i8>, W_q: &mut Array3<i8>, B_q: &mut Array3<i8>, bias: u8) {

    println!("===================== Query Projection Space Transformation =====================");

    if bias == 1 {

        for i in 0..Q_p.shape()[0] {
            // Loop over the number of heads
            for j in 0..Q_p.shape()[1] {
                // Loop over the number of queries
                for k in 0..Q_p.shape()[2] {
                    Q_p[[i, j, k]] = B_q[[i, j, k]] as i32;
                    // Loop over the number of features
                    for l in 0..Q.shape()[1] {
                        Q_p[[i, j, k]] += Q[[j, l]] as i32 * W_q[[i, l, k]] as i32;
                    }
                }
            }
        }
        
    } else {
        // Loop over the number of heads
        for i in 0..Q_p.shape()[0] {
            // Loop over the number of queries
            for j in 0..Q_p.shape()[1] {
                // Loop over the number of keys
                for k in 0..Q_p.shape()[2] {
                    Q_p[[i, j, k]] = 0;
                    // Loop over the number of features
                    for l in 0..Q.shape()[1] {
                        Q_p[[i, j, k]] += Q[[j, l]] as i32 * W_q[[i, l, k]] as i32;
                    }
                }
            }
        }
    }

    println!("Q_p: {:?}", Q_p);
}

pub fn key_projection_space_transformation(K_p: &mut Array3<i32>, K: &mut Array2<i8>, W_k: &mut Array3<i8>, B_k: &mut Array3<i8>, bias: u8) {

    println!("===================== Key Projection Space Transformation =====================");

    if bias == 1 {

        for i in 0..K_p.shape()[0] {
            // Loop over the number of heads
            for j in 0..K_p.shape()[1] {
                // Loop over the number of queries
                for k in 0..K_p.shape()[2] {
                    K_p[[i, j, k]] = B_k[[i, j, k]] as i32;
                    // Loop over the number of features
                    for l in 0..K.shape()[1] {
                        K_p[[i, j, k]] += K[[j, l]] as i32 * W_k[[i, l, k]] as i32;
                    }
                }
            }
        }
        
    } else {
        // Loop over the number of heads
        for i in 0..K_p.shape()[0] {
            // Loop over the number of queries
            for j in 0..K_p.shape()[1] {
                // Loop over the number of keys
                for k in 0..K_p.shape()[2] {
                    K_p[[i, j, k]] = 0;
                    // Loop over the number of features
                    for l in 0..K.shape()[1] {
                        K_p[[i, j, k]] += K[[j, l]] as i32 * W_k[[i, l, k]] as i32;
                    }
                }
            }
        }
    }

    println!("K_p: {:?}", K_p);
}

pub fn value_projection_space_transformation(V_p: &mut Array3<i32>, V: &mut Array2<i8>, W_v: &mut Array3<i8>, B_v: &mut Array3<i8>, bias: u8) {

    println!("===================== Value Projection Space Transformation =====================");

    if bias == 1 {

        for i in 0..V_p.shape()[0] {
            // Loop over the number of heads
            for j in 0..V_p.shape()[1] {
                // Loop over the number of queries
                for k in 0..V_p.shape()[2] {
                    V_p[[i, j, k]] = B_v[[i, j, k]] as i32;
                    // Loop over the number of features
                    for l in 0..V.shape()[1] {
                        V_p[[i, j, k]] += V[[j, l]] as i32 * W_v[[i, l, k]] as i32;
                    }
                }
            }
        }
        
    } else {
        // Loop over the number of heads
        for i in 0..V_p.shape()[0] {
            // Loop over the number of queries
            for j in 0..V_p.shape()[1] {
                // Loop over the number of keys
                for k in 0..V_p.shape()[2] {
                    V_p[[i, j, k]] = 0;
                    // Loop over the number of features
                    for l in 0..V.shape()[1] {
                        V_p[[i, j, k]] += V[[j, l]] as i32 * W_v[[i, l, k]] as i32;
                    }
                }
            }
        }
    }

    println!("V_p: {:?}", V_p);
}

pub fn query_key_correlation(Qp_requant: &mut Array3<i8>, Kp_requant: &mut Array3<i8>, QK: &mut Array3<i32>) {

    println!("===================== Query Key Correlation =====================");

    // Loop over the number of heads
    for i in 0..QK.shape()[0] {
        // Loop over the number of queries
        for j in 0..QK.shape()[1] {
            // Loop over the number of keys
            for k in 0..QK.shape()[2] {
                QK[[i, j, k]] = 0;
                // Loop over the number of features
                for l in 0..QK.shape()[1] {
                    QK[[i, j, k]] += Qp_requant[[i, j, l as usize]] as i32 * Kp_requant[[i, k, l as usize]] as i32;
                }
            }
        }
    }

    println!("QK: {:?}", QK);
}

//Compute the approximated softmax function.
pub fn streaming_partial_softmax(A_requant: &mut Array3<i8>, A_partial_softmax: &mut Array2<i32>, seq_len: i32) {

    println!("===================== Streaming Partial SoftMax =====================");
    
    let log2e: f64 = f64::log2(f64::exp(1.0));
    let x = Array::linspace(-255f32, 0.0, 256);
    let b = 8;
    let eps_x = b as f64 / (2.0f64.powi(b) * log2e);
    let mut exp_partial_sum = Array1::<i32>::zeros(seq_len as usize);
    let mut max = Array1::<i8>::zeros(64);
    let mut current_max = Array1::<i8>::zeros(64);
    
    for i in 0..4 {
        let A_requant_slice = A_requant.slice_mut(s![0, .., i * 16..(i + 1) * 16]);
        
        for n in 0..A_requant_slice.nrows() {
            current_max[[n]] = A_requant_slice.row(n).iter().copied().max().unwrap() as i8;
        }

        for j in 0..seq_len {
            let mut shift_sum;
            if i==0 || current_max[j as usize] > max[[j as usize]] {
                if i==0 {
                    shift_sum = 0;
                } else {
                    shift_sum = (current_max[j as usize]-max[[j as usize]])/32; 
                    if ((((current_max[j as usize]-max[[j as usize]])/32))-shift_sum) as f64 >= 0.5 {
                        shift_sum += 1;
                    }
                }
                max[j as usize] = current_max[j as usize];
            } else {
                shift_sum = 0;
            }

            let qb = A_requant.slice_mut(s![0, .., i*16..(i+1)*16]).mapv(|x| x - max[[j as usize]]);
            
            let mut qexp = 0;
            for k in 0..qb.ncols() {
                let mut shift = (-qb[[j as usize,k]]) as i32 /32;
                let shift_int = (-qb[[j as usize,k]]) as i32;

                if shift_int % 32 >= 16 {
                    shift += 1;
                }

                qexp += (2_u32.pow(10) >> shift as i32) as i32;
            }

            exp_partial_sum[[j as usize]] = (exp_partial_sum[[j as usize]] >> shift_sum as i32) + qexp;
        }
    }
    for j in 0..seq_len {
        let factor = (((2.0f64.powi(8) - 1.0) * 2.0f64.powi(10)) as i32 / exp_partial_sum[j as usize]);
        for k in 0..seq_len {
            
            let mut shift = ((max[j as usize]-(A_requant[[0, j as usize,k as usize]]))/32) as i32;
            let shift_int = max[j as usize]-(A_requant[[0, j as usize,k as usize]]) as i8;
            if shift_int % 32 >= 16 {
                shift += 1;
            }
            A_partial_softmax[[j as usize,k as usize]] = (factor as i32) / 2.0f64.powi(shift) as i32;
        }
    }

    println!("A_partial_softmax: {}", A_partial_softmax);
}

pub fn single_head_computation(A_partial_softmax: &mut Array2<i32>, Vp_requant: &mut Array3<i8>, O_softmax: &mut Array3<i32>) {

    println!("===================== Single Head Computation =====================");

    // Loop over the number of heads
    for i in 0..O_softmax.shape()[0] {
        // Loop over the number of queries
        for j in 0..O_softmax.shape()[1] {
            // Loop over the number of keys
            for k in 0..O_softmax.shape()[2] {
                O_softmax[[i, j, k]] = 0;
                // Loop over the number of features
                for l in 0..O_softmax.shape()[1] {
                    O_softmax[[i, j, k]] += A_partial_softmax[[j, l]] as i32 * Vp_requant[[i, l, k]] as i32;
                }
            }
        }
    }

    println!("O_softmax: {:?}", O_softmax);
    
}
