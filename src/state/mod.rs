use std::fmt::Debug;

mod macros;
pub(crate) use macros::*;

/// Mask for extracting borrowed bit (bit 0)
pub const BORROW_MASK: usize = 0b1;

#[cfg(not(feature = "weak"))]
mod flags {
    /// Mask for reference count. It's also the maximum reference count (RC_MAX)
    /// we can have, since the last bit is used for borrow flags.
    pub const RC_MASK: usize = !0b1;

    /// One unit of reference count (bit 1 set)
    pub const RC_UNIT: usize = 0b10;
}

#[cfg(feature = "weak")]
mod flags {
    use super::*;

    /// Number of shifts allocated for strong reference count.
    pub const STRONG_SHIFT: u32 = usize::BITS / 2;

    /// Mask for extracting strong count from the state.
    pub const STRONG_MASK: usize = usize::MAX << STRONG_SHIFT;

    /// One unit of strong reference count.
    pub const STRONG_UNIT: usize = 1 << STRONG_SHIFT;

    /// Mask for extracting weak count from the state.
    pub const WEAK_MASK: usize = !(STRONG_MASK | BORROW_MASK);

    /// One unit of weak reference count.
    pub const WEAK_UNIT: usize = 0b10;
}

#[cfg(feature = "weak")]
pub use flags::*;
#[cfg(not(feature = "weak"))]
pub use flags::*;

/// Snapshot of the current state.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Snapshot(pub usize);

impl Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(not(feature = "weak"))]
        {
            f.debug_struct("Snapshot")
                .field("count", &self.count())
                .field("borrowed", &self.is_borrowed())
                .finish()
        }
        #[cfg(feature = "weak")]
        {
            f.debug_struct("Snapshot")
                .field("strong", &self.strong_count())
                .field("weak", &self.weak_count())
                .field("borrowed", &self.is_borrowed())
                .finish()
        }
    }
}

#[cfg(not(feature = "weak"))]
impl Snapshot {
    /// Current reference count.
    pub fn count(&self) -> usize {
        (self.0 & RC_MASK) >> 1
    }

    pub fn is_shared(&self) -> bool {
        self.count() > 1
    }

    pub fn is_borrowed(&self) -> bool {
        (self.0 & BORROW_MASK) != 0
    }
}

#[cfg(feature = "weak")]
impl Snapshot {
    /// Alias for [`strong_count`]
    ///
    /// [`strong_count`]: Self::strong_count
    pub fn count(&self) -> usize {
        self.strong_count()
    }

    /// Current strong reference count.
    pub fn strong_count(&self) -> usize {
        (self.0 & STRONG_MASK) >> STRONG_SHIFT
    }

    /// Current weak reference count.
    pub fn weak_count(&self) -> usize {
        (self.0 & WEAK_MASK) >> 1
    }

    pub fn is_shared(&self) -> bool {
        self.strong_count() > 1
    }

    pub fn is_borrowed(&self) -> bool {
        (self.0 & BORROW_MASK) != 0
    }
}

impl From<usize> for Snapshot {
    fn from(value: usize) -> Self {
        Snapshot(value)
    }
}
