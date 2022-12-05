mod softmax;
use softmax::
{
    softmax_test,
    soft_max_s8,
};

fn main() {
    softmax_test();
    let data_in = [1,2,3,4,5];
    let data_out: Vec<i8> = vec![0; 5]; 
    soft_max_s8(&data_in, data_out,1,1,1,1,1,1,1);
    println!("Hello, world!");
}