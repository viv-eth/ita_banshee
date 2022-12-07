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

//Compute the approximated softmax function.
pub fn streaming_partial_softmax(A_requant: &mut Array2<i32>, A_partial_softmax: &mut Array2<i32>, seq_len: i32) {

    println!("===================== Streaming Partial SoftMax =====================");
    
    let log2e: f64 = f64::log2(f64::exp(1.0));
    let x = Array::linspace(-255f32, 0.0, 256);
    let b = 8;
    let eps_x = b as f64 / (2.0f64.powi(b) * log2e);
    let mut exp_partial_sum = Array1::<i32>::zeros(seq_len as usize);
    let mut max = Array1::<i8>::zeros(64);
    let mut current_max = Array1::<i8>::zeros(64);
    let mut max_test: i32 = 0;
    for i in 0..4 {
        let A_requant_slice = A_requant.slice_mut(s![.., i * 16..(i + 1) * 16]);
        
        for i in 0..A_requant_slice.nrows() {
            current_max[[i]] = A_requant_slice.row(i).iter().copied().max().unwrap() as i8;
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
                    max[j as usize] = current_max[j as usize];
                }
            } else {
                shift_sum = 0;
            }
            let qb = A_requant.slice_mut(s![.., i*16..(i+1)*16]).mapv(|x| x - (max[[j as usize]]) as i32);
            // println!("qb: {}", qb);
            // println!("qb shape: {:?}", qb.shape());
            let mut qexp = 0;
            for k in 0..qb.ncols() {
                let mut shift = (-qb[[j as usize,k]]) as i32 /32;
                let shift_int = (-qb[[j as usize,k]]) as i32;
                // TODO: Why is this 0.5???
                // if ((-qb[[j as usize,k]]/32) - shift)  >= 0.5 {
                //     shift += 1;
                // }
                if shift_int % 32 >= 16 {
                    shift += 1;
                }
                qexp += (2.0f64.powi(10) / 2.0f64.powi(shift)) as i32;
            }
            exp_partial_sum[[j as usize]] = (exp_partial_sum[[j as usize]] >> shift_sum as i32) + qexp;
        }
    }
    for j in 0..seq_len {
        let factor = (((2.0f64.powi(8) - 1.0) * 2.0f64.powi(10)) as i32 / exp_partial_sum[j as usize]);
        for k in 0..seq_len {
            // What is going on with the types???
            let mut shift = ((max[j as usize]-(A_requant[[j as usize,k as usize]])as i8)/32) as i32;
            let shift_int = max[j as usize]-(A_requant[[j as usize,k as usize]])as i8;
            if shift_int % 32 >= 16 {
                shift += 1;
            }
            A_partial_softmax[[j as usize,k as usize]] = (factor as i32) / 2.0f64.powi(shift) as i32;
        }
    }

    println!("A_partial_softmax: {}", A_partial_softmax);
}
