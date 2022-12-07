extern crate ndarray;
extern crate rand;
extern crate round;
extern crate npy;

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
    streaming_partial_softmax,
};

fn main() {
    let mut A_requant = Array2::<i32>::zeros((64, 64));
    let mut Q = Array2::<i8>::zeros((64, 64));
    let mut W_q = Array3::<i8>::zeros((1, 64, 64));
    let mut B_q = Array3::<i8>::zeros((1, 64, 64));
    let mut A_partial_softmax = Array2::<i32>::zeros((64, 64));
    let mut A_temp = Array1::<i32>::zeros((64 * 64));
    let mut Q_temp = Array1::<i8>::zeros((64 * 64));
    let mut W_q_temp = Array1::<i8>::zeros((1 * 64 * 64));
    let mut B_q_temp = Array1::<i8>::zeros((1 * 64 * 64));

    // Init A_requant with random numbers
    // for i in 0..64 {
    //     for j in 0..64 {
    //         A_requant[[i, j]] = rand::thread_rng().gen_range(-128..127);
    //     }
    // }

    let mut A_requant_buf = vec![]; 
    std::fs::File::open("/scratch/vivianep/ita_mempool/ita/Python_model/A_requant.npy").unwrap()
        .read_to_end(&mut A_requant_buf).unwrap();
    
    let A_requant_matrix: NpyData<i64> = NpyData::from_bytes(&A_requant_buf).unwrap();

    // let A_requant_matrix: NpyData<i64> = NpyData::from_bytes(&buf).unwrap();

    let mut cnt = 0;

    for number in A_requant_matrix {
        A_temp[cnt] = number as i32;
        cnt += 1;
    }

    // instantiate A_requant with data from A_temp
    for i in 0..A_requant.shape()[0] {
        for j in 0..A_requant.shape()[1] {
            A_requant[[i, j]] = A_temp[i * 64 + j];
        }
    }

    let mut Q_buf = vec![]; 
    std::fs::File::open("/scratch/vivianep/ita_mempool/ita/Python_model/Q_matrix.npy").unwrap()
        .read_to_end(&mut Q_buf).unwrap();

    // let Q_data: NpyData<i8> = NpyData::from_bytes(&Q_buf).unwrap();
    let Q_matrix: NpyData<i64> = NpyData::from_bytes(&Q_buf).unwrap();

    // convert Q_matrix to i8
    let mut Q_cnt = 0;
    for number in Q_matrix {
        Q_temp[Q_cnt] = number as i8;
        Q_cnt += 1;
    }

    // instantiate Q with data from Q_temp
    for i in 0..Q.shape()[0] {
        for j in 0..Q.shape()[1] {
            Q[[i, j]] = Q_temp[i * 64 + j]
        }
    }

    let mut W_q_buf = vec![];
    std::fs::File::open("/scratch/vivianep/ita_mempool/ita/Python_model/Wq_matrix.npy").unwrap()
        .read_to_end(&mut W_q_buf).unwrap();

    let W_q_matrix: NpyData<i64> = NpyData::from_bytes(&W_q_buf).unwrap();

    let mut W_q_cnt = 0;

    for number in W_q_matrix {
        W_q_temp[W_q_cnt] = number as i8;
        W_q_cnt += 1;
    }

    // instantiate W_q with data from W_q_temp
    for i in 0..W_q.shape()[0] {
        for j in 0..W_q.shape()[1] {
            for k in 0..W_q.shape()[2] {
                W_q[[i, j, k]] = W_q_temp[i * 64 * 64 + j * 64 + k];
            }
        }
    }
    
    streaming_partial_softmax(&mut A_requant, &mut A_partial_softmax, 64);
    // println!("A_requant: {}", A_requant);
    // println!("A_partial_softmax: {}", A_partial_softmax);
    println!("A_partial_softmax shape: {:?}", A_partial_softmax.shape());
    println!("Q shape: {:?}", Q.shape());
    println!("W_q shape: {:?}", W_q.shape());
    println!("B_q shape: {:?}", B_q.shape());
    println!("Hello, world!");
    // println!("cnt: {}", cnt);
}