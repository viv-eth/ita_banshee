// TODO: implement functions for matrix
// initialization

use ndarray::{prelude::*, Array2};
use npy::NpyData;
use std::io::Read;
// RUST compiler throws a warning for this include
// but it is necessary!!
use std::any::type_name;

use num::cast::AsPrimitive;

// Instantiate 2D matrix from npy file

pub fn init2D_matrix<T: npy::Serializable + Copy + 'static + AsPrimitive<T> + std::fmt::Debug>(
    m: &mut Array2<T>,
    m_path: &str,
) where
    i8: AsPrimitive<T>,
    i32: AsPrimitive<T>,
{
    // Read buffer for npy file
    let mut m_buf = vec![];

    // Read npy file and data in little endian
    std::fs::File::open(m_path)
        .unwrap()
        .read_to_end(&mut m_buf)
        .unwrap();

    let m_data: NpyData<i32> = NpyData::from_bytes(&m_buf).unwrap();

    // get dimensions of m
    let m_shape = m.shape();

    let m_dim1 = m_shape[0];

    // write data in m
    for (i, number) in m_data.into_iter().enumerate() {
        m[[i / m_dim1, i % m_dim1]] = number.as_();
    }
}

// Instantiate 3D matrix from npy file

pub fn init3D_matrix<T: npy::Serializable + Copy + 'static + AsPrimitive<T> + std::fmt::Debug>(
    m: &mut Array3<T>,
    m_path: &str,
) where
    i8: AsPrimitive<T>,
    i32: AsPrimitive<T>,
{
    // Read buffer for npy file
    let mut m_buf = vec![];

    // Read npy file and data in little endian
    std::fs::File::open(m_path)
        .unwrap()
        .read_to_end(&mut m_buf)
        .unwrap();

    let m_data: NpyData<i32> = NpyData::from_bytes(&m_buf).unwrap();

    // get dimensions of m
    let m_shape = m.shape();

    let m_dim1 = m_shape[1];
    let m_dim2 = m_shape[2];

    // write data in m
    for (i, number) in m_data.into_iter().enumerate() {
        m[[
            i / (m_dim1 * m_dim2),
            (i % (m_dim1 * m_dim2)) / m_dim1,
            i % m_dim1,
        ]] = number.as_();
    }
}
