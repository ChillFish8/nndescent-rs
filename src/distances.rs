use std::fmt::{Debug, Formatter};

use crate::Array;


#[derive(Copy, Clone)]
/// The given
pub enum Distance {
    Euclidean,
    SquaredEuclidean,
    Cosine,
    AlternativeCosine,
    Dot,
    AlternativeDot,
    Custom(fn(&Array, &Array) -> f32),
}

impl Debug for Distance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Distance::Euclidean => write!(f, "Euclidean"),
            Distance::SquaredEuclidean => write!(f, "SquaredEuclidean"),
            Distance::Cosine => write!(f, "Cosine"),
            Distance::AlternativeCosine => write!(f, "AlternativeCosine"),
            Distance::Dot => write!(f, "Dot"),
            Distance::AlternativeDot => write!(f, "AlternativeDot"),
            Distance::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl Distance {
    #[inline(always)]
    /// Calculates the distance between two arrays.
    pub fn get_distance(&self, left: &Array, right: &Array) -> f32 {
        match self {
            Self::Euclidean => left.euclidean(right),
            Self::SquaredEuclidean => left.squared_euclidean(right),
            Self::Cosine => left.cosine(right),
            Self::AlternativeCosine => left.alternative_cosine(right),
            Self::Dot => left.dot_adjusted(right),
            Self::AlternativeDot => left.alternative_dot_adjusted(right),
            Self::Custom(cb) => cb(left, right),
        }
    }
}
