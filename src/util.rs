// TODO: implement functions for matrix
// initialization

use ndarray:: {
    prelude::*,
    Array2,
};
use npy::NpyData;
use std::io::Read;
use std::any::type_name;

use num::cast::AsPrimitive;

// Instantiate 2D matrix from npy file

pub fn init2D_matrix<T: npy::Serializable + Copy + 'static + AsPrimitive<T> + std::fmt::Debug>(M: &mut Array2<T>, M_path: &str) where i8: AsPrimitive<T>, i32: AsPrimitive<T> {

    // Read buffer for npy file
    let mut M_buf = vec![];

    // Read npy file and data in little endian
    std::fs::File::open(M_path).unwrap()
        .read_to_end(&mut M_buf).unwrap();
    
    let M_data: NpyData<i32> = NpyData::from_bytes(&M_buf).unwrap();

    // get dimensions of M
    let m_shape = M.shape();

    let m_dim1 = m_shape[0];


    // write data in M
    for (i, number) in M_data.into_iter().enumerate() {
        M[[i / m_dim1, i % m_dim1]] = number.as_();
    }
}

// Instantiate 3D matrix from npy file

pub fn init3D_matrix<T: npy::Serializable + Copy + 'static + AsPrimitive<T> + std::fmt::Debug>(M: &mut Array3<T>, M_path: &str) where i8: AsPrimitive<T>, i32: AsPrimitive<T> {

    // Read buffer for npy file
    let mut M_buf = vec![];

    // Read npy file and data in little endian
    std::fs::File::open(M_path).unwrap()
        .read_to_end(&mut M_buf).unwrap();

    let M_data: NpyData<i32> = NpyData::from_bytes(&M_buf).unwrap();

    // get dimensions of M
    let m_shape = M.shape();

    let m_dim1 = m_shape[0];
    let m_dim2 = m_shape[1];

    // write data in M
    for (i, number) in M_data.into_iter().enumerate() {
        M[[i / (m_dim1 * m_dim2), (i % (m_dim1 * m_dim2)) / m_dim1, i % m_dim1]] = number.as_();
    }
}