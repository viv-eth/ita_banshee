extern crate ndarray;
extern crate npy;
extern crate num;
extern crate rand;
extern crate round;

use ndarray::{prelude::*, Array2};

mod softmax;
use softmax::{
    projection_space_transformation, multi_head_computation, parallel_requantize3d,
    query_key_correlation, requantization_3d,
    single_head_computation, streaming_partial_softmax,
};

mod util;
use util::{init2D_matrix, init3D_matrix};

fn main() {
    // Setup of matrices for query_projection_space_transformation and key_projection_space_transformation
    let mut q = Array2::<i8>::zeros((64, 64));
    init2D_matrix(
        &mut q,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Q_matrix.npy",
    );
    let mut w_q = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut w_q,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Wq_matrix.npy",
    );

    let mut k = Array2::<i8>::zeros((64, 64));
    init2D_matrix(
        &mut k,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/K_matrix.npy",
    );
    let mut w_k = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut w_k,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Wk_matrix.npy",
    );

    // Setup of matrices for value_projection_space_transformation
    let mut b_v = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut b_v,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Bv_matrix.npy",
    );
    let mut v = k.clone();
    let mut w_v = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut w_v,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Wv_matrix.npy",
    );
    let mut v_p = Array3::<i32>::zeros((1, 64, 64));

    // matrices in the query_projection_space_transformation
    let mut b_q = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut b_q,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Bq_matrix.npy",
    );
    let mut q_p = Array3::<i32>::zeros((1, 64, 64));

    // matrices in the key_projection_space_transformation
    let mut b_k = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut b_k,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Bk_matrix.npy",
    );
    let mut k_p = Array3::<i32>::zeros((1, 64, 64));

    // matrices in the streaming_partial_softmax
    let mut a_requant = Array3::<i8>::zeros((1, 64, 64));
    let mut a_partial_softmax = Array2::<i32>::zeros((64, 64));

    // matrices in multi_head_computation
    let mut out = Array3::<i32>::zeros((1, 64, 64));
    let mut b_o = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut b_o,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Bo_matrix.npy",
    );
    let mut w_o = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(
        &mut w_o,
        "/scratch/vivianep/ita_mempool/ita/Python_model/matrices/Wo_matrix.npy",
    );

    // query_projection_space_transformation
    // query_projection_space_transformation(&mut q_p, &mut q, &mut w_q, &mut b_q, 1);
    projection_space_transformation(&mut q_p, &mut q, &mut w_q, &mut b_q, 1);
    // requantization of q_p
    let mut q_p_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut q_p, &mut q_p_requant, 52, 14);
    println!("q_p_requant: {}", q_p_requant);

    // key_projection_space_transformation
    // key_projection_space_transformation(&mut k_p, &mut k, &mut w_k, &mut b_k, 1);
    projection_space_transformation(&mut k_p, &mut k, &mut w_k, &mut b_k, 1);
    // requantization of k_p
    let mut k_p_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut k_p, &mut k_p_requant, 66, 14);
    println!("k_p_requant: {}", k_p_requant);

    // query_key_correlation
    let mut qk = Array3::<i32>::zeros((1, 64, 64));
    query_key_correlation(&mut q_p_requant, &mut k_p_requant, &mut qk);
    // requantization of qk
    requantization_3d(&mut qk, &mut a_requant, 19, 14);
    println!("a_requant: {}", a_requant);

    // streaming_partial_softmax
    streaming_partial_softmax(&mut a_requant, &mut a_partial_softmax, 64);

    // value_projection_space_transformation
    // value_projection_space_transformation(&mut v_p, &mut v, &mut w_v, &mut b_v, 1);
    projection_space_transformation(&mut v_p, &mut v, &mut w_v, &mut b_v, 1);
    // requantization of v_p
    let mut v_p_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut v_p, &mut v_p_requant, 54, 14);
    println!("v_p_requant: {}", v_p_requant);

    // single_head_computation
    let mut o_softmax = Array3::<i32>::zeros((1, 64, 64));
    single_head_computation(&mut a_partial_softmax, &mut v_p_requant, &mut o_softmax);
    // requantization of o_softmax
    let mut o_softmax_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut o_softmax, &mut o_softmax_requant, 76, 14);
    println!("o_softmax_requant: {}", o_softmax_requant);

    // multi_head_computation
    multi_head_computation(&mut o_softmax_requant, &mut out, &mut w_o, &mut b_o, 1);
    // parallel requantization of out
    let mut out_requant = Array2::<i8>::zeros((64, 64));
    parallel_requantize3d(&mut out, &mut out_requant, 6, 14);
    println!("out_requant: {}", out_requant);
}
