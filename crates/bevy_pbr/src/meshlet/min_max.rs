use core::simd::{cmp::SimdPartialEq, f32x4, num::SimdFloat};
use std::simd::StdFloat;

pub struct Normalizer {
    min: f32,
    mul: f32x4,
    add: f32x4,
    denorm_scale: f32x4,
}

const ONE: f32x4 = f32x4::from_array([1.; 4]);
const TWO: f32x4 = f32x4::from_array([2.; 4]);
const ZERO: f32x4 = f32x4::from_array([0.; 4]);

impl Normalizer {
    #[inline(always)]
    pub fn normalize(&self, position: &[f32; 3]) -> [f32; 3] {
        let [x, y, z] = *position;
        let v = f32x4::from_array([x, y, z, 0.])
            .mul_add(self.mul, self.add)
            .simd_clamp(f32x4::splat(self.min), ONE);
        [v[0], v[1], v[2]]
    }

    #[inline(always)]
    pub fn denorm_scale(&self) -> [f32; 3] {
        let [x, y, z, _] = self.denorm_scale.to_array();
        [x, y, z]
    }

    #[inline(always)]
    pub fn denorm_scale_normalized_position(&self, normalized_position: &[f32; 3]) -> [f32; 3] {
        let [x, y, z] = *normalized_position;
        let v = f32x4::from_array([x, y, z, 0.]) * self.denorm_scale;
        [v[0], v[1], v[2]]
    }
}

#[derive(Copy, Clone)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct MinMax {
    pub min: f32x4,
    pub max: f32x4,
}

impl MinMax {
    #[inline]
    pub fn new() -> Self {
        Self {
            min: f32x4::splat(f32::MAX),
            max: f32x4::splat(f32::MIN),
        }
    }

    #[inline]
    pub fn update(&mut self, v: &[f32; 3]) {
        let v = f32x4::from_array([v[0], v[1], v[2], 0.]);
        self.min = self.min.simd_min(v);
        self.max = self.max.simd_max(v);
    }

    #[inline]
    pub fn delta(&self) -> f32x4 {
        self.max - self.min
    }

    #[inline]
    pub fn max_delta_component(&self) -> f32x4 {
        f32x4::splat(self.delta().reduce_max())
    }

    #[inline]
    pub fn normalization_multiply_add(&self) -> (f32x4, f32x4) {
        let delta = self.delta();
        let zero_mask = delta.simd_eq(ZERO);

        // Derivation of Multiplication and Addition constants
        //
        // Let v be a value in the range of [min, max].
        // Let delta be max - min.
        // Let v' be the value of v in the normalized range [0,1].
        //
        // v' = ((v - min) / delta)
        // v' = (v/delta - min/delta)
        // v' = (v * (1/delta) + -min/delta)
        //           ^-------^   ^--------^
        //              mul         add
        let n_m = zero_mask.select(ONE, delta.recip());
        let n_a = zero_mask.select(-self.min, -self.min / delta);
        (n_m, n_a)
    }

    #[inline]
    pub fn normalizer(&self) -> Normalizer {
        let (mul, add) = self.normalization_multiply_add();
        let denorm_scale = self.delta() / self.max_delta_component();
        Normalizer {
            min: 0.,
            mul,
            add,
            denorm_scale,
        }
    }

    #[inline]
    pub fn signed_normalization_multiply_add(&self) -> (f32x4, f32x4) {
        let delta = self.delta();
        let zero_mask = delta.simd_eq(ZERO);

        // Derivation of Multiplication and Addition constants
        //
        // Let v be a value in the range of [min, max].
        // Let delta be max - min.
        // Let v' be the value of v in the normalized range [-1,1].
        //
        // v' = (v - (min + delta/2)) * 2/delta
        // v' = v * 2/delta - (2min/delta + 1)
        // v' = v * 2/delta + -(2min/delta + 1)
        // v' = v * 2/delta + -2min/delta - 1
        //          ^-----^   ^-------------^
        //             mul         add
        let n_m = zero_mask.select(ONE, TWO / delta);
        let n_a = zero_mask.select(-(TWO * self.min) - ONE, -(TWO * self.min / delta) - ONE);
        (n_m, n_a)
    }

    #[inline]
    pub fn signed_normalizer(&self) -> Normalizer {
        let (mul, add) = self.signed_normalization_multiply_add();
        let denorm_scale = self.delta() / self.max_delta_component();
        Normalizer {
            min: -1.,
            mul,
            add,
            denorm_scale,
        }
    }
}
