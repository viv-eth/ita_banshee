
pub fn softmax_test() {
    println!("Hello from softmax!");
}

pub fn soft_max_s8(data_in: &[i8], mut data_out: Vec<i8>, size: i32, last_dim_length: i32, 
                  coeff_a: i32, coeff_b: i32, coeff_c: i32, log2: i32, n_levels: u32) {

    let (mut x_tilde, mut z, mut p): (i8, i8, i8);
    let mut x_max: i8;
    let mut y_sum: u32;
    let mut y: Vec<u32> = vec![0; last_dim_length as usize]; 

    for i in 0..(size / last_dim_length) {
        y_sum = 0;
        x_max = -128;

        for j in 0..last_dim_length {
            if data_in[(j+i*last_dim_length) as usize] > x_max {
                x_max = data_in[(j+i*last_dim_length) as usize];
            }
        }

        for j in 0..last_dim_length {
            x_tilde = data_in[(j+i*last_dim_length) as usize] - x_max;
            z = (- (x_tilde as i32) / log2) as i8;
            p = ((x_tilde as i32) + (z as i32) * log2) as i8;

            y[j as usize] = ((coeff_a*(((p as i32)+coeff_b)*((p as i32)+coeff_b)) + coeff_c)>>(z)) as u32;
            y_sum += y[j as usize];
        }

        for j in 0..last_dim_length {
            data_out[(j + i*last_dim_length) as usize] = ((y[j as usize]*(n_levels-1))/(y_sum) - n_levels/2) as i8;
        }
    }
}