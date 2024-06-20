use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct TriangularMatrix<T> {
    pub size: usize,
    pub arr: ndarray::Array1<T>
}

impl<T: std::fmt::Debug> std::fmt::Debug for TriangularMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //f.debug_struct("TriangularMatrix").field("vec", &self.vec).field("size", &self.size).finish()
        let mut rows = vec![Vec::with_capacity(self.size); self.size];
        for i in 0..self.size {
            for j in 0..=i {
                rows[i].push(format!("{:?}", self[[i, j]]));
            }
        }
        let max_len = rows.iter().fold(0, |acc, x| acc.max(x.iter().fold(0, |acc2, y| acc2.max(y.len()))));
        for r in rows.iter_mut() {
            for c in r.iter_mut() {
                *c = format!("{:>1$}", c, max_len);
            }
        }
        let m: Vec<String> = rows.iter().map(|x| x.join("  ")).collect();
        f.write_str(&m.join("\n"))
    }
}

pub fn triangular_matrix_len(n: usize) -> usize {
    (n * (n + 1)).div_euclid(2)
}

pub fn triangular_matrix_ij(index: usize) -> (usize, usize) {
    let i = ((1usize + 8usize*index).isqrt()-1usize).div_euclid(2);
    let j = index - triangular_matrix_index(i, 0);
    (i, j)
}

pub fn triangular_matrix_index(i: usize, j: usize) -> usize {
    let small = i.min(j);
    let big = i.max(j);
    (big * (big + 1)).div_euclid(2) + small
}

impl<T: Clone> TriangularMatrix<T> {
    pub fn row(&self, index: usize) -> Array1<T> {
        //(0..self.size).map(|i| &self.arr[triangular_matrix_index(index, i)]).collect()
        Array1::from_iter((0..self.size).map(|i| self.arr[triangular_matrix_index(index, i)].clone()))
    }

    pub fn diag(&self) -> Vec<&T> {
        (0..self.size).map(|i| &self.arr[triangular_matrix_index(i, i)]).collect()
    }

    pub fn diag_mut(&mut self) -> Vec<&mut T> {
        let mut v = Vec::with_capacity(self.size);
        let mut offset = 0;
        let mut arr_next = self.arr.as_slice_mut().unwrap();
        for i in 0..self.size {
            let arr_prev;
            let idx = triangular_matrix_index(i, i) - offset;
            (arr_prev, arr_next) = arr_next.split_at_mut(idx+1);
            offset += idx + 1;
            v.push(arr_prev.last_mut().unwrap());
        }
        v
    }


    pub fn row_mut(&mut self, index: usize) -> Vec<&mut T> {
        let mut v = Vec::with_capacity(self.size);
        let mut offset = 0;
        let mut arr_next= self.arr.as_slice_mut().unwrap();
        for i in 0..self.size {
            let arr_prev;
            let idx = triangular_matrix_index(index, i) - offset;
            (arr_prev, arr_next) = arr_next.split_at_mut(idx+1);
            offset += idx + 1;
            v.push(arr_prev.last_mut().unwrap());
        }
        v
    }
}

impl<T: Clone> TriangularMatrix<T> {
    pub fn drop(&self, index: usize) -> TriangularMatrix<T> {
        TriangularMatrix {
            size: self.size,
            arr: self.arr.iter().enumerate().filter(|(idx, _x)| {
                let (i, j) = triangular_matrix_ij(*idx);
                (i == index) || (j == index)
            }).map(|(_idx, x)| x.clone()).collect()
        }
    }
}

impl<T> std::ops::Index<[usize; 2]> for TriangularMatrix<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let idx = triangular_matrix_index(index[0], index[1]);
        &self.arr[idx]
    }
}

impl<T> std::ops::IndexMut<[usize; 2]> for TriangularMatrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let idx = triangular_matrix_index(index[0], index[1]);
        self.arr.index_mut(idx)
    }
}


impl<T: Clone> TriangularMatrix<T> {
    pub fn fill(n: usize, value: T) -> Self {
        TriangularMatrix {
            size: n,
            arr: ndarray::Array1::from_vec(vec![value; triangular_matrix_len(n)])
        }
    }
}

impl<T: Default + Clone> TriangularMatrix<T> {
    pub fn zeros(n: usize) -> Self {
        TriangularMatrix {
            size: n,
            arr: ndarray::Array1::from_vec(vec![T::default(); triangular_matrix_len(n)])
        }
    }
}

impl<T: std::ops::Neg + Clone> std::ops::Neg for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Neg>::Output>;
    
    fn neg(self) -> Self::Output {
        TriangularMatrix {
            arr: self.arr.iter().map(|x| -x.clone()).collect(),
            size: self.size
        }
    }
}

impl<T, U> std::ops::Add<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Add<Array1<T>, Output = Array1<U>>
{
    type Output = TriangularMatrix<U>;

    fn add(self, rhs: Self) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::add(self.arr, rhs.arr),
            size: self.size
        }
    }
}

impl<T, U, V> std::ops::Add<U> for TriangularMatrix<T>
where
    T: std::ops::Add<U, Output = V>,
    Array1<T>: std::ops::Add<U, Output = Array1<V>>,
{
    type Output = TriangularMatrix<V>;

    fn add(self, rhs: U) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::add(self.arr, rhs),
            size: self.size
        }
    }
}

impl<T> std::ops::AddAssign<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::AddAssign
{
    fn add_assign(&mut self, rhs: Self) {
        ndarray::Array1::add_assign(&mut self.arr, rhs.arr);
    }
}

impl<T, U> std::ops::AddAssign<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::AddAssign<U>,
{

    fn add_assign(&mut self, rhs: U) {
        self.arr += rhs
    }
}

impl<T, U> std::ops::Sub<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Sub<Array1<T>, Output = Array1<U>>
{
    type Output = TriangularMatrix<U>;

    fn sub(self, rhs: Self) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::sub(self.arr, rhs.arr),
            size: self.size
        }
    }
}

impl<T, U, V> std::ops::Sub<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Sub<U, Output = Array1<V>>,
{
    type Output = TriangularMatrix<V>;

    fn sub(self, rhs: U) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::sub(self.arr, rhs),
            size: self.size
        }
    }
}

impl<T> std::ops::SubAssign<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::SubAssign
{
    fn sub_assign(&mut self, rhs: Self) {
        ndarray::Array1::sub_assign(&mut self.arr, rhs.arr);
    }
}

impl<T, U> std::ops::SubAssign<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::SubAssign<U>,
{

    fn sub_assign(&mut self, rhs: U) {
        ndarray::Array1::sub_assign(&mut self.arr, rhs);
    }
}

impl<T, U> std::ops::Mul<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Mul<Array1<T>, Output = Array1<U>>
{
    type Output = TriangularMatrix<U>;

    fn mul(self, rhs: Self) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::mul(self.arr, rhs.arr),
            size: self.size
        }
    }
}

impl<T, U, V> std::ops::Mul<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Mul<U, Output = Array1<V>>,
{
    type Output = TriangularMatrix<V>;

    fn mul(self, rhs: U) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::mul(self.arr, rhs),
            size: self.size
        }
    }
}

impl<T> std::ops::MulAssign<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::MulAssign
{
    fn mul_assign(&mut self, rhs: Self) {
        ndarray::Array1::mul_assign(&mut self.arr, rhs.arr);
    }
}

impl<T, U> std::ops::MulAssign<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::MulAssign<U>,
{

    fn mul_assign(&mut self, rhs: U) {
        ndarray::Array1::mul_assign(&mut self.arr, rhs);
    }
}
impl<T, U> std::ops::Div<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Div<Array1<T>, Output = Array1<U>>
{
    type Output = TriangularMatrix<U>;

    fn div(self, rhs: Self) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::div(self.arr, rhs.arr),
            size: self.size
        }
    }
}

impl<T, U, V> std::ops::Div<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::Div<U, Output = Array1<V>>,
{
    type Output = TriangularMatrix<V>;

    fn div(self, rhs: U) -> Self::Output {
        TriangularMatrix {
            arr: ndarray::Array1::div(self.arr, rhs),
            size: self.size
        }
    }
}

impl<T> std::ops::DivAssign<Self> for TriangularMatrix<T>
where
    Array1<T>: std::ops::DivAssign
{
    fn div_assign(&mut self, rhs: Self) {
        ndarray::Array1::div_assign(&mut self.arr, rhs.arr);
    }
}

impl<T, U> std::ops::DivAssign<U> for TriangularMatrix<T>
where
    Array1<T>: std::ops::DivAssign<U>,
{

    fn div_assign(&mut self, rhs: U) {
        ndarray::Array1::div_assign(&mut self.arr, rhs);
    }
}


impl<T: Clone> TriangularMatrix<T> {
    pub fn convert<U: From<T>>(self) -> TriangularMatrix<U> {
        TriangularMatrix {
            arr: self.arr.iter().map(|x| x.clone().into()).collect(),
            size: self.size,
        }
    }
    
    pub fn try_convert<U: TryFrom<T>>(self) -> Result<TriangularMatrix<U>, <U as TryFrom<T>>::Error> {
        Ok(TriangularMatrix {
            arr: self.arr.iter().map(|x| U::try_from(x.clone())).try_collect()?,
            size: self.size,
        })
    }
    
    pub fn map<U, F: FnMut(&T) -> U>(self, mapper: F) -> TriangularMatrix<U> {
        TriangularMatrix {
            arr: self.arr.iter().map(mapper).collect(),
            size: self.size,
        }
    }
}

impl TriangularMatrix<f32> {
    pub fn fillna(&mut self, val: f32) {
        for v in self.arr.iter_mut() {
            if v.is_nan() {
                *v = val;
            }
        }
    }
}

impl TriangularMatrix<f64> {
    pub fn fillna(&mut self, val: f64) {
        for v in self.arr.iter_mut() {
            if v.is_nan() {
                *v = val;
            }
        }
    }
}