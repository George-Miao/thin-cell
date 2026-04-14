macro_rules! impl_state {
    {
        $( #[$meta:meta] )*
        struct State($inner:ident);
    } => {
        use std::{process::abort, sync::atomic::Ordering::*};

        /// Encapsulates the bitwise logic for Reference Counting and borrow flags.
        ///
        /// All bits except last are used for Reference Count (RC), while last bit is
        /// used for borrow flags (Borrowed).
        ///
        $( #[$meta] )*
        #[repr(transparent)]
        pub struct State($inner);

        impl std::fmt::Debug for State {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_tuple("State").field(&self.load()).finish()
            }
        }

        impl State {
            #[cfg(not(feature = "weak"))]
            pub fn new() -> Self {
                // Starts with 1 owner, 0 borrows
                State($inner::new(RC_UNIT))
            }

            #[cfg(feature = "weak")]
            pub fn new() -> Self {
                // Starts with 1 strong owner, 1 implicit weak ref, 0 borrows.
                // The implicit weak ref is held collectively by all strong refs
                // and released when the last strong ref is dropped.
                State($inner::new(STRONG_UNIT | WEAK_UNIT))
            }

            pub fn load(&self) -> Snapshot {
                self.0.load(Relaxed).into()
            }

            #[cfg(not(feature = "weak"))]
            pub fn inc(&self) -> &Self {
                // As explained in `Arc`'s comment, use relaxed ordering is fine for
                // reference count increment.
                let orig = self.0.fetch_add(RC_UNIT, Relaxed);

                // Quote unquote from `Arc`:
                // > This branch will never be taken in any realistic program. We abort because
                // > such a program is incredibly degenerate, and we don't care to support it.
                if (orig & RC_MASK) == RC_MASK {
                    abort()
                }

                self
            }

            #[cfg(feature = "weak")]
            pub fn inc(&self) -> &Self {
                // Increment strong reference count
                let orig = self.0.fetch_add(STRONG_UNIT, Relaxed);

                // Check for overflow in strong count
                if (orig & STRONG_MASK) == STRONG_MASK {
                    abort()
                }

                self
            }

            #[cfg(feature = "weak")]
            pub fn inc_weak(&self) -> &Self {
                // Increment weak reference count
                let orig = self.0.fetch_add(WEAK_UNIT, Relaxed);

                // Check for overflow in weak count
                if (orig & WEAK_MASK) == WEAK_MASK {
                    abort()
                }

                self
            }

            /// Try to increment strong count only if it's non-zero.
            /// Returns true if successful, false if strong count is zero.
            #[cfg(feature = "weak")]
            pub fn try_inc(&self) -> bool {
                let mut current = self.0.load(Relaxed);

                loop {
                    // Check if strong count is zero
                    if (current & STRONG_MASK) == 0 {
                        return false;
                    }

                    // Try to increment strong count
                    let new = current + STRONG_UNIT;

                    // Check for overflow
                    if (new & STRONG_MASK) == 0 {
                        abort()
                    }

                    match self.0.compare_exchange_weak(current, new, Relaxed, Relaxed) {
                        Ok(_) => return true,
                        Err(actual) => current = actual,
                    }
                }
            }

            /// Decrease reference count by one.
            ///
            /// Returns whether reference count has reached zero (needs drop).
            #[cfg(not(feature = "weak"))]
            pub fn dec(&self) -> bool {
                // Because `fetch_sub` is already atomic, we do not need to synchronize
                // with other threads unless we are going to delete the object.
                if self.0.fetch_sub(RC_UNIT, Release) != RC_UNIT {
                    return false;
                }

                debug_assert!(
                    !self.load().is_borrowed(),
                    "Reference count should never reach zero while borrowed"
                );

                // Prevent any other thread from reading after we have decremented the
                // count to zero, which could lead to use-after-free.
                self.acquire();

                true
            }

            #[cfg(feature = "weak")]
            pub fn dec(&self) -> bool {
                // Decrement strong reference count
                let prev = self.0.fetch_sub(STRONG_UNIT, Release);

                // Extract just the strong count from prev
                let prev_strong = (prev & STRONG_MASK);

                // If we weren't the last strong reference, don't drop
                if prev_strong != STRONG_UNIT {
                    return false;
                }

                debug_assert!(
                    !self.load().is_borrowed(),
                    "Strong count should never reach zero while borrowed"
                );

                // Synchronize before drop
                self.acquire();

                true
            }

            #[cfg(feature = "weak")]
            pub fn dec_weak(&self) -> bool {
                // Decrement weak reference count
                let prev = self.0.fetch_sub(WEAK_UNIT, Release);

                // Extract weak and strong counts from prev
                let prev_weak = (prev & WEAK_MASK);
                let prev_strong = (prev & STRONG_MASK);

                // Only deallocate if both strong and weak are zero after decrement
                if prev_weak != WEAK_UNIT || prev_strong != 0 {
                    return false;
                }

                // Synchronize before deallocation
                self.acquire();

                true
            }
            /// Try to zero the reference count if there is only one owner and not borrowed.
            ///
            /// Returns whether the unwrapping is successful (i.e., we can safely take the underlying object).
            #[cfg(not(feature = "weak"))]
            pub fn try_unwrap(&self) -> bool {
                // Only when `state == RC_UNIT` (one owner, not borrowed) can we safely unwrap.
                // Any other state means either multiple owners or borrowed, both of which
                // prevent unwrapping.
                if self
                    .0
                    .compare_exchange(RC_UNIT, 0, Release, Relaxed)
                    .is_err()
                {
                    return false;
                }

                // Similar to `dec`, we need to synchronize with other threads to prevent them from reading the
                // object after we have took it.
                self.acquire();

                true
            }

            #[cfg(feature = "weak")]
            pub fn try_unwrap(&self) -> bool {
                let mut current = self.0.load(Relaxed);

                loop {
                    if (current & STRONG_MASK) != STRONG_UNIT || (current & BORROW_MASK) != 0 {
                        return false;
                    }

                    let new = current - STRONG_UNIT;

                    match self.0.compare_exchange_weak(current, new, Release, Relaxed) {
                        Ok(_) => break,
                        Err(actual) => current = actual,
                    }
                }

                // Synchronize with other threads
                self.acquire();

                true
            }

            #[cfg(not(feature = "weak"))]
            pub fn unborrow(&self) {
                // Keep RC bits, clear Borrow bits
                self.0.fetch_and(RC_MASK, Release);
            }

            #[cfg(feature = "weak")]
            pub fn unborrow(&self) {
                // Clear borrow bit, keep strong and weak counts
                self.0.fetch_and(!BORROW_MASK, Release);
            }
        }
    };
}

macro_rules! test_cases {
    ($usize:ty) => {
        #[test]
        fn test_state_new() {
            let state = State::new().load();
            assert_eq!(state.count(), 1);
            assert!(!state.is_borrowed());
            assert!(!state.is_shared());
        }

        #[test]
        fn test_state_count() {
            let state = State::new();
            assert_eq!(state.load().count(), 1);

            state.inc();
            assert_eq!(state.load().count(), 2);

            state.inc();
            assert_eq!(state.load().count(), 3);
        }

        #[test]
        fn test_state_inc() {
            let state = State::new();
            assert_eq!(state.load().count(), 1);

            for i in 2..=10 {
                state.inc();
                assert_eq!(state.load().count(), i);
            }
        }

        #[test]
        fn test_state_dec() {
            let state = State::new();
            state.inc();
            state.inc();
            state.inc(); // count = 4

            state.dec(); // count = 3
            assert_eq!(state.load().count(), 3);

            state.dec(); // count = 2
            assert_eq!(state.load().count(), 2);

            state.dec(); // count = 1
            assert_eq!(state.load().count(), 1);
        }

        #[test]
        fn test_state_is_shared() {
            let state = State::new();
            assert!(!state.load().is_shared());

            state.inc();
            assert!(state.load().is_shared());

            state.inc();
            assert!(state.load().is_shared());

            state.dec();
            assert!(state.load().is_shared());

            state.dec();
            assert!(!state.load().is_shared());
        }

        #[test]
        fn test_state_borrow() {
            let state = State::new();
            assert!(!state.load().is_borrowed());

            state.borrow();
            let borrowed = state.load();
            assert!(borrowed.is_borrowed());
            assert_eq!(borrowed.count(), 1); // Strong count unchanged
        }

        #[test]
        fn test_state_try_borrow_success() {
            let state = State::new();
            let success = state.try_borrow();
            assert!(success);

            let borrowed = state.load();
            assert!(borrowed.is_borrowed());
            assert_eq!(borrowed.count(), 1); // Strong count unchanged
        }

        #[test]
        fn test_state_try_borrow_failure() {
            let state = State::new();
            state.borrow();

            // Already borrowed, should fail
            let success = state.try_borrow();
            assert!(!success);
        }

        #[test]
        fn test_state_unborrow() {
            let state = State::new();
            state.borrow();
            assert!(state.load().is_borrowed());

            state.unborrow();
            let unborrowed = state.load();
            assert!(!unborrowed.is_borrowed());
            assert_eq!(unborrowed.count(), state.load().count()); // Strong count unchanged
        }

        #[test]
        fn test_state_borrow_with_multiple_refs() {
            let state = State::new();
            state.inc();
            state.inc(); // count = 3
            assert!(!state.load().is_borrowed());

            state.borrow();
            let borrowed = state.load();
            assert!(borrowed.is_borrowed());
            assert_eq!(borrowed.count(), 3); // Strong count unchanged

            state.unborrow();
            let unborrowed = state.load();
            assert!(!unborrowed.is_borrowed());
            assert_eq!(unborrowed.count(), 3);
        }

        #[test]
        fn test_state_borrow_preserves_rc() {
            let state = State::new();
            state.inc();
            state.inc(); // count = 3
            let original_count = state.load().count();

            state.borrow();
            assert_eq!(state.load().count(), original_count);

            state.unborrow();
            assert_eq!(state.load().count(), original_count);
        }

        #[test]
        fn test_state_eq() {
            let state1 = State::new();
            let state2 = State::new();
            assert_eq!(state1.load(), state2.load());

            state1.inc();
            assert_ne!(state1.load(), state2.load());

            state1.dec();
            assert_eq!(state1.load(), state2.load());

            state1.borrow();
            assert_ne!(state1.load(), state2.load());
        }

        #[test]
        #[cfg(feature = "weak")]
        fn test_weak_count() {
            let state = State::new();
            assert_eq!(state.load().weak_count(), 1);
            assert_eq!(state.load().strong_count(), 1);

            state.inc_weak();
            assert_eq!(state.load().weak_count(), 2);
            assert_eq!(state.load().strong_count(), 1);

            state.inc_weak();
            assert_eq!(state.load().weak_count(), 3);
            assert_eq!(state.load().strong_count(), 1);
        }

        #[test]
        #[cfg(feature = "weak")]
        fn test_weak_inc_dec() {
            let state = State::new();

            state.inc_weak();
            state.inc_weak();
            state.inc_weak(); // 3 weak refs
            assert_eq!(state.load().weak_count(), 4);

            state.dec_weak();
            assert_eq!(state.load().weak_count(), 3);

            state.dec_weak();
            assert_eq!(state.load().weak_count(), 2);

            state.dec_weak();
            assert_eq!(state.load().weak_count(), 1);
        }

        #[test]
        #[cfg(feature = "weak")]
        fn test_strong_and_weak_independent() {
            let state = State::new();

            // Add strong refs
            state.inc();
            state.inc(); // 3 strong
            assert_eq!(state.load().strong_count(), 3);
            assert_eq!(state.load().weak_count(), 1);

            // Add weak refs
            state.inc_weak();
            state.inc_weak(); // 2 weak
            assert_eq!(state.load().strong_count(), 3);
            assert_eq!(state.load().weak_count(), 3);

            // Remove a strong ref
            state.dec();
            assert_eq!(state.load().strong_count(), 2);
            assert_eq!(state.load().weak_count(), 3);

            // Remove a weak ref
            state.dec_weak();
            assert_eq!(state.load().strong_count(), 2);
            assert_eq!(state.load().weak_count(), 2);
        }

        #[test]
        #[cfg(feature = "weak")]
        fn test_weak_with_borrow() {
            let state = State::new();
            state.inc_weak();
            state.inc_weak(); // 2 weak, 1 strong

            assert!(!state.load().is_borrowed());

            state.borrow();
            let borrowed = state.load();
            assert!(borrowed.is_borrowed());
            assert_eq!(borrowed.strong_count(), 1);
            assert_eq!(borrowed.weak_count(), 3);

            state.unborrow();
            let unborrowed = state.load();
            assert!(!unborrowed.is_borrowed());
            assert_eq!(unborrowed.strong_count(), 1);
            assert_eq!(unborrowed.weak_count(), 3);
        }

        #[test]
        #[cfg(feature = "weak")]
        fn test_try_unwrap_with_weak() {
            let state = State::new();

            // With only the collective strong-side weak ref, should succeed.
            assert!(state.try_unwrap());
            assert_eq!(state.load().strong_count(), 0);
            assert_eq!(state.load().weak_count(), 1);

            // Reset and add an explicit weak ref.
            let state = State::new();
            state.inc_weak();

            // Explicit weak refs do not block unwrapping.
            assert!(state.try_unwrap());
            assert_eq!(state.load().strong_count(), 0);
            assert_eq!(state.load().weak_count(), 2);
        }

        #[test]
        #[cfg(feature = "weak")]
        fn test_dec_weak_deallocate() {
            let state = State::new();
            state.inc_weak(); // 1 strong, 1 explicit weak (+1 implicit weak)

            // Drop strong ref first
            let should_drop_value = state.dec();
            assert!(should_drop_value); // Value should be dropped
            assert_eq!(state.load().strong_count(), 0);
            assert_eq!(state.load().weak_count(), 2);

            // Drop the implicit weak ref held by the strong side.
            let should_deallocate = state.dec_weak();
            assert!(!should_deallocate);
            assert_eq!(state.load().strong_count(), 0);
            assert_eq!(state.load().weak_count(), 1);

            // Drop the remaining explicit weak ref - should signal deallocation.
            let should_deallocate = state.dec_weak();
            assert!(should_deallocate); // Should deallocate memory
        }
    };
}

pub(crate) use impl_state;
pub(crate) use test_cases;
