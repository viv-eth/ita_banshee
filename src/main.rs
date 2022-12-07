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
    streaming_partial_softmax,
    query_projection_space_transformation,
};

mod util;
use util::
{
    init2D_matrix,
    init3D_matrix,
};

fn main() {

    // matrices in the streaming_partial_softmax
    let mut A_requant = Array2::<i32>::zeros((64, 64));
    let mut A_partial_softmax = Array2::<i32>::zeros((64, 64));
    let mut Q = Array2::<i8>::zeros((64, 64));
    let mut W_q = Array3::<i8>::zeros((1, 64, 64));
    
    // matrices in the query_projection_space_transformation
    let mut B_q = Array3::<i8>::zeros((1, 64, 64));
    let mut Q_p = Array3::<i32>::zeros((1, 64, 64));

    // temporary matrices
    let mut A_temp = Array1::<i32>::zeros((64 * 64));

    // Init A_requant with random numbers
    // for i in 0..64 {
    //     for j in 0..64 {
    //         A_requant[[i, j]] = rand::thread_rng().gen_range(-128..127);
    //     }
    // }

    let mut A_requant_buf = vec![]; 
    std::fs::File::open("/scratch/vivianep/ita_mempool/ita/Python_model/A_requant.npy").unwrap()
        .read_to_end(&mut A_requant_buf).unwrap();
    
    let A_requant_matrix: NpyData<i32> = NpyData::from_bytes(&A_requant_buf).unwrap();

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

    init2D_matrix(&mut Q, "/scratch/vivianep/ita_mempool/ita/Python_model/Q_matrix.npy");

    init3D_matrix(&mut W_q, "/scratch/vivianep/ita_mempool/ita/Python_model/Wq_matrix.npy");

    init3D_matrix(&mut B_q, "/scratch/vivianep/ita_mempool/ita/Python_model/Bq_matrix.npy");

    println!("W_q: {}", W_q);
    println!("B_q: {}", B_q);
    println!("Q: {}", Q);

    query_projection_space_transformation(&mut Q_p, &mut Q, &mut W_q, &mut B_q, 1);
    
    // streaming_partial_softmax(&mut A_requant, &mut A_partial_softmax, 64);
    // println!("A_requant: {}", A_requant);
    // println!("A_partial_softmax: {}", A_partial_softmax);
    println!("A_partial_softmax shape: {:?}", A_partial_softmax.shape());
    println!("Q shape: {:?}", Q.shape());
    println!("W_q shape: {:?}", W_q.shape());
    println!("B_q shape: {:?}", B_q.shape());
    println!("Hello, world!");
    // println!("cnt: {}", cnt);
}