use std::iter::zip;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct TriangularMatrix<T> {
    pub size: usize,
    pub vec: Vec<T>
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

pub fn triangular_matrix_len(n: usize) -> usize {
    (n * (n + 1)).div_euclid(2)
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

impl<T> TriangularMatrix<T> {
    pub fn row(&self, index: usize) -> Vec<&T> {
        let mut v = Vec::with_capacity(self.size);
        for i in 0..self.size {
            v.push(&self.vec[triangular_matrix_index(index, i)]);
        }
        v
    }

    pub fn row_mut(&mut self, index: usize) -> Vec<&mut T> {
        let mut v = Vec::with_capacity(self.size);
        let mut offset = 0;
        let mut arr_next: &mut [T] = &mut self.vec;
        for i in 0..self.size {
            let arr_prev;
            let idx = triangular_matrix_index(index, i) - offset;
            (arr_prev, arr_next) = arr_next.split_at_mut(idx+1);
            offset += idx;
            v.push(arr_prev.last_mut().unwrap());
        }
        v
    }
}

impl<T: Clone> TriangularMatrix<T> {
    pub fn drop(&self, index: usize) -> TriangularMatrix<T> {
        TriangularMatrix {
            size: self.size,
            vec: self.vec.iter().enumerate().filter(|(idx, _x)| {
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
        &self.vec[idx]
    }
}

impl<T> std::ops::IndexMut<[usize; 2]> for TriangularMatrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let idx = triangular_matrix_index(index[0], index[1]);
        self.vec.index_mut(idx)
    }
}


impl<T: Clone> TriangularMatrix<T> {
    pub fn fill(n: usize, value: T) -> Self {
        TriangularMatrix {
            size: n,
            vec: vec![value; triangular_matrix_len(n)]
        }
    }
}

impl<T: Default + Clone> TriangularMatrix<T> {
    pub fn zeros(n: usize) -> Self {
        TriangularMatrix {
            size: n,
            vec: vec![T::default(); triangular_matrix_len(n)]
        }
    }
}

impl<T: std::ops::Neg + Clone> std::ops::Neg for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Neg>::Output>;
    
    fn neg(self) -> Self::Output {
        TriangularMatrix {
            vec: self.vec.iter().map(|x| -x.clone()).collect(),
            size: self.size
        }
    }
}

impl<T: std::ops::Add + Clone> std::ops::Add<Self> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Add<T>>::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.vec.len() == rhs.vec.len(), "cannot add matrices of different shape");
        let mut v = Vec::with_capacity(self.vec.len());
        for (a, b) in zip(self.vec.iter(), rhs.vec.iter()) {
            let c = a.clone() + b.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Add + Clone> std::ops::Add<T> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Add<T>>::Output>;

    fn add(self, rhs: T) -> Self::Output {
        let mut v = Vec::with_capacity(self.vec.len());
        for a in self.vec.iter() {
            let c = a.clone() + rhs.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Add<Output = T> + Clone> std::ops::AddAssign<Self> for TriangularMatrix<T> {
    fn add_assign(&mut self, rhs: Self) {
        for (a, b) in zip(self.vec.iter_mut(), rhs.vec.iter()) {
            let c: T = a.clone() + b.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Add<Output = T> + Clone> std::ops::AddAssign<T> for TriangularMatrix<T> {
    fn add_assign(&mut self, rhs: T) {
        for a in self.vec.iter_mut() {
            let c: T = a.clone() + rhs.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Sub + Clone> std::ops::Sub<Self> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Sub<T>>::Output>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.vec.len() == rhs.vec.len(), "cannot subtract matrices of different shape");
        let mut v = Vec::with_capacity(self.vec.len());
        for (a, b) in zip(self.vec.iter(), rhs.vec.iter()) {
            let c = a.clone() - b.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Sub + Clone> std::ops::Sub<T> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Sub<T>>::Output>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut v = Vec::with_capacity(self.vec.len());
        for a in self.vec.iter() {
            let c = a.clone() - rhs.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Sub<Output = T> + Clone> std::ops::SubAssign<Self> for TriangularMatrix<T> {
    fn sub_assign(&mut self, rhs: Self) {
        for (a, b) in zip(self.vec.iter_mut(), rhs.vec.iter()) {
            let c: T = a.clone() - b.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Sub<Output = T> + Clone> std::ops::SubAssign<T> for TriangularMatrix<T> {
    fn sub_assign(&mut self, rhs: T) {
        for a in self.vec.iter_mut() {
            let c: T = a.clone() - rhs.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Mul + Clone> std::ops::Mul<Self> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Mul<T>>::Output>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.vec.len() == rhs.vec.len(), "cannot multiply matrices of different shape");
        let mut v = Vec::with_capacity(self.vec.len());
        for (a, b) in zip(self.vec.iter(), rhs.vec.iter()) {
            let c = a.clone() * b.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Mul + Clone> std::ops::Mul<T> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Mul<T>>::Output>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut v = Vec::with_capacity(self.vec.len());
        for a in self.vec.iter() {
            let c = a.clone() * rhs.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Mul<Output = T> + Clone> std::ops::MulAssign<Self> for TriangularMatrix<T> {
    fn mul_assign(&mut self, rhs: Self) {
        for (a, b) in zip(self.vec.iter_mut(), rhs.vec.iter()) {
            let c: T = a.clone() * b.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Mul<Output = T> + Clone> std::ops::MulAssign<T> for TriangularMatrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        for a in self.vec.iter_mut() {
            let c: T = a.clone() * rhs.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Div + Clone> std::ops::Div<Self> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Div<T>>::Output>;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(self.vec.len() == rhs.vec.len(), "cannot divide matrices of different shape");
        let mut v = Vec::with_capacity(self.vec.len());
        for (a, b) in zip(self.vec.iter(), rhs.vec.iter()) {
            let c = a.clone() / b.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Div + Clone> std::ops::Div<T> for TriangularMatrix<T> {
    type Output = TriangularMatrix<<T as std::ops::Div<T>>::Output>;

    fn div(self, rhs: T) -> Self::Output {
        let mut v = Vec::with_capacity(self.vec.len());
        for a in self.vec.iter() {
            let c = a.clone() / rhs.clone();
            v.push(c);
        }
        TriangularMatrix {
            vec: v,
            size: self.size
        }
    }
}

impl<T: std::ops::Div<Output = T> + Clone> std::ops::DivAssign<Self> for TriangularMatrix<T> {
    fn div_assign(&mut self, rhs: Self) {
        for (a, b) in zip(self.vec.iter_mut(), rhs.vec.iter()) {
            let c: T = a.clone() / b.clone();
            *a = c;
        }
    }
}

impl<T: std::ops::Div<Output = T> + Clone> std::ops::DivAssign<T> for TriangularMatrix<T> {
    fn div_assign(&mut self, rhs: T) {
        for a in self.vec.iter_mut() {
            let c: T = a.clone() / rhs.clone();
            *a = c;
        }
    }
}

impl<T: Clone> TriangularMatrix<T> {
    pub fn convert<U: From<T>>(self) -> TriangularMatrix<U> {
        TriangularMatrix {
            vec: self.vec.iter().map(|x| x.clone().into()).collect(),
            size: self.size,
        }
    }
    
    pub fn try_convert<U: TryFrom<T>>(self) -> Result<TriangularMatrix<U>, <U as TryFrom<T>>::Error> {
        Ok(TriangularMatrix {
            vec: self.vec.iter().map(|x| U::try_from(x.clone())).try_collect()?,
            size: self.size,
        })
    }
    
    pub fn map<U, F: FnMut(&T) -> U>(self, mapper: F) -> TriangularMatrix<U> {
        TriangularMatrix {
            vec: self.vec.iter().map(mapper).collect(),
            size: self.size,
        }
    }
}

impl TriangularMatrix<f32> {
    pub fn fillna(&mut self, val: f32) {
        for v in self.vec.iter_mut() {
            if v.is_nan() {
                *v = val;
            }
        }
    }
}

impl TriangularMatrix<f64> {
    pub fn fillna(&mut self, val: f64) {
        for v in self.vec.iter_mut() {
            if v.is_nan() {
                *v = val;
            }
        }
    }
}
