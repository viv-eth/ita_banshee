extern crate ndarray;
extern crate rand;
extern crate round;
extern crate npy;
extern crate num;

use ndarray:: {
    prelude::*,
    Array2,
};
use npy::NpyData;
use std::io::Read;

use round::{
    round, 
    round_up, 
    round_down};

use rand::Rng;

mod softmax;
use softmax::
{
    requantization_3d,
    query_projection_space_transformation,
    key_projection_space_transformation,
    value_projection_space_transformation,
    query_key_correlation,
    streaming_partial_softmax,
    single_head_computation,
};

mod util;
use util::
{
    init2D_matrix,
    init3D_matrix,
};

fn main() {
    
    // Setup of matrices for query_projection_space_transformation and key_projection_space_transformation
    let mut Q = Array2::<i8>::zeros((64, 64));
    init2D_matrix(&mut Q, "/scratch/vivianep/ita_mempool/ita/Python_model/Q_matrix.npy");
    let mut W_q = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(&mut W_q, "/scratch/vivianep/ita_mempool/ita/Python_model/Wq_matrix.npy");
    
    let mut K = Array2::<i8>::zeros((64, 64));
    init2D_matrix(&mut K, "/scratch/vivianep/ita_mempool/ita/Python_model/K_matrix.npy");
    let mut W_k = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(&mut W_k, "/scratch/vivianep/ita_mempool/ita/Python_model/Wk_matrix.npy");

    // Setup of matrices for value_projection_space_transformation
    let mut B_v = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(&mut B_v, "/scratch/vivianep/ita_mempool/ita/Python_model/Bv_matrix.npy");
    let mut V = K.clone();
    let mut W_v = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(&mut W_v, "/scratch/vivianep/ita_mempool/ita/Python_model/Wv_matrix.npy");
    let mut V_p = Array3::<i32>::zeros((1, 64, 64));
    
    // matrices in the query_projection_space_transformation
    let mut B_q = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(&mut B_q, "/scratch/vivianep/ita_mempool/ita/Python_model/Bq_matrix.npy");
    let mut Q_p = Array3::<i32>::zeros((1, 64, 64));
    
    
    // matrices in the key_projection_space_transformation
    let mut B_k = Array3::<i8>::zeros((1, 64, 64));
    init3D_matrix(&mut B_k, "/scratch/vivianep/ita_mempool/ita/Python_model/Bk_matrix.npy");
    let mut K_p = Array3::<i32>::zeros((1, 64, 64));

    // matrices in the streaming_partial_softmax
    let mut A_requant = Array3::<i8>::zeros((1, 64, 64));
    let mut A_partial_softmax = Array2::<i32>::zeros((64, 64));

    
    println!("W_q: {}", W_q);
    println!("B_q: {}", B_q);
    println!("Q: {}", Q);

    println!("W_k: {}", W_k);
    println!("B_k: {}", B_k);
    println!("K: {}", K);



    // query_projection_space_transformation
    query_projection_space_transformation(&mut Q_p, &mut Q, &mut W_q, &mut B_q, 1);
    // requantization of Q_p
    let mut Q_p_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut Q_p, &mut Q_p_requant, 52, 14);
    println!("Q_p_requant: {}", Q_p_requant);

    // key_projection_space_transformation
    key_projection_space_transformation(&mut K_p, &mut K, &mut W_k, &mut B_k, 1);
    // requantization of K_p
    let mut K_p_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut K_p, &mut K_p_requant, 66, 14);
    println!("K_p_requant: {}", K_p_requant);

    // query_key_correlation
    let mut QK = Array3::<i32>::zeros((1, 64, 64));
    query_key_correlation(&mut Q_p_requant, &mut K_p_requant, &mut QK);
    // requantization of QK
    requantization_3d(&mut QK, &mut A_requant, 19, 14);
    println!("A_requant: {}", A_requant);

    // streaming_partial_softmax
    streaming_partial_softmax(&mut A_requant, &mut A_partial_softmax, 64);

    // value_projection_space_transformation
    value_projection_space_transformation(&mut V_p, &mut V, &mut W_v, &mut B_v, 1);
    // requantization of V_p
    let mut V_p_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut V_p, &mut V_p_requant, 54, 14);
    println!("V_p_requant: {}", V_p_requant);

    // single_head_computation
    let mut O_softmax = Array3::<i32>::zeros((1, 64, 64));
    single_head_computation(&mut A_partial_softmax, &mut V_p_requant, &mut O_softmax);
    // requantization of O_softmax
    let mut O_softmax_requant = Array3::<i8>::zeros((1, 64, 64));
    requantization_3d(&mut O_softmax, &mut O_softmax_requant, 76, 14);
    println!("O_softmax_requant: {}", O_softmax_requant);

}