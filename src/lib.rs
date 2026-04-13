#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod state;

mod fat_ptr;

pub mod sync;
pub mod unsync;

macro_rules! thin_cell {
    {
        $( #[$doc:meta] )*
    } => {
        use std::{
            any::{Any, TypeId},
            cell::UnsafeCell,
            fmt::{self, Debug, Display},
            marker::PhantomData,
            mem::ManuallyDrop,
            ops::{Deref, DerefMut},
            ptr::NonNull,
        };

        use crate::fat_ptr::*;

        /// The inner allocation of `ThinCell`
        ///
        /// This should not be used except in unsize coercion solely as a type.
        #[repr(C)]
        pub struct Inner<T: ?Sized> {
            // metadata MUST be at offset 0 so that `*mut Inner<T>` is also a valid `*mut usize`
            // points to the metadata
            metadata: usize,
            state: State,
            data: UnsafeCell<T>,
        }

        $( #[$doc] )*
        pub struct ThinCell<T: ?Sized> {
            ptr: NonNull<()>,
            _marker: PhantomData<Inner<T>>,
        }

        /// A mutable guard returned by [`ThinCell::borrow`]
        pub struct Ref<'a, T: ?Sized> {
            value: &'a mut T,
            state: &'a State,
        }

        /// A weak reference to a [`ThinCell`] that doesn't prevent dropping.
        ///
        /// `Weak` references don't keep the value alive. The value will be dropped
        /// when all strong references (`ThinCell`) are dropped, even if weak
        /// references still exist.
        ///
        /// Use [`Weak::upgrade`] to attempt to convert a weak reference back to a
        /// strong reference (`ThinCell`). This will fail if the value has already
        /// been dropped.
        #[cfg(feature = "weak")]
        pub struct Weak<T: ?Sized> {
            ptr: NonNull<()>,
            _marker: PhantomData<Inner<T>>,
        }

        impl<T> ThinCell<T> {
            /// Creates a new `ThinCell` wrapping the given data.
            pub fn new(data: T) -> Self {
                let alloc = Box::new(Inner {
                    metadata: 0,
                    state: State::new(),
                    data: UnsafeCell::new(data),
                });

                let ptr = Box::into_raw(alloc);

                ThinCell {
                    ptr: unsafe { NonNull::new_unchecked(ptr as _) },
                    _marker: PhantomData,
                }
            }

            /// Consumes the `ThinCell` and try to get inner value.
            ///
            /// Returns the inner value in [`Ok`] if there are no other owners and it is
            /// not currently borrowed, return `Err(self)` otherwise.
            pub fn try_unwrap(self) -> Result<T, Self> {
                if !self.inner().state.try_unwrap() {
                    return Err(self);
                }

                // SAFETY: As tested above, there are no other owners and it is not borrowed
                Ok(unsafe { self.unwrap_unchecked() })
            }

            /// Consumes the `ThinCell`, returning the inner value.
            ///
            /// # Safety
            /// The caller must guarantee that there are no other owners and it is not
            /// currently borrowed.
            #[cfg(not(feature = "weak"))]
            pub unsafe fn unwrap_unchecked(self) -> T {
                let this = ManuallyDrop::new(self);
                // SAFETY: guaranteed by caller to have unique ownership and is not borrowed
                let inner = unsafe { Box::from_raw(this.inner_ptr() as *mut Inner<T>) };

                inner.data.into_inner()
            }

            /// Consumes the `ThinCell`, returning the inner value.
            ///
            /// # Safety
            /// The caller must guarantee that there are no other strong owners and it is not
            /// currently borrowed.
            #[cfg(feature = "weak")]
            pub unsafe fn unwrap_unchecked(self) -> T {
                let this = ManuallyDrop::new(self);
                let inner = this.inner();

                let _weak: Weak<T> = Weak {
                    ptr: this.ptr,
                    _marker: PhantomData,
                };

                // SAFETY: guaranteed by caller to have unique strong ownership and
                // not be borrowed. This moves out `T` without touching the
                // allocation so outstanding weak refs remain valid.
                unsafe { std::ptr::read(inner.data.get()) }
            }
        }

        impl<T: ?Sized> ThinCell<T> {
            const IS_SIZED: bool = is_sized::<T>();

            /// Reconstructs the raw pointer to the inner allocation.
            fn inner_ptr(&self) -> *const Inner<T> {
                let ptr = self.ptr.as_ptr();

                if Self::IS_SIZED {
                    // SIZED CASE: Cast pointer-to-pointer
                    // Doing this trick to workaround Rust not allowing `ptr as *const Inner<T>`
                    // due to `T` being `?Sized` directly even when we know it's `Sized`
                    let ptr_ref = &ptr as *const *mut () as *const *const Inner<T>;

                    // SAFETY: `self.ptr` is a valid pointer of `Inner<T>`
                    unsafe { *ptr_ref }
                } else {
                    // UNSIZED CASE: Read metadata
                    // SAFETY: pointer returned by `self.ptr` is valid, and `metadata` is at offset
                    // 0 of `Inner<T>`, which is guaranteed by `repr(C)` and the definition of
                    // `Inner<T>`
                    let metadata = unsafe { *(ptr as *const usize) };

                    // Miri will complain about this:
                    // - https://github.com/thepowersgang/stack_dst-rs/issues/14
                    // - https://github.com/uazu/stakker/blob/5821c30409c19ca9167808b669c928c94bc5f177/src/queue/flat.rs#L14-L17
                    // But this should be sound as per Rust's fat pointer and metadata construction
                    FatPtr { ptr, metadata }.into_ptr()
                }
            }

            /// Returns a reference to the inner allocation.
            fn inner(&self) -> &Inner<T> {
                unsafe { &*self.inner_ptr() }
            }

            /// Returns a reference to the state cell.
            fn state(&self) -> &State {
                &self.inner().state
            }

            /// Deallocates the inner allocation.
            ///
            /// # Safety
            ///
            /// `self` must be the last owner and it must not be used after this call.
            #[cfg(not(feature = "weak"))]
            unsafe fn drop_in_place(&mut self) {
                drop(unsafe { Box::from_raw(self.inner_ptr() as *mut Inner<T>) })
            }

            /// Leaks the `ThinCell`, returning a raw pointer to the inner allocation.
            ///
            /// The returned pointer points to the inner allocation. To restore the
            /// `ThinCell`, use [`ThinCell::from_raw`].
            pub fn leak(self) -> *mut () {
                let this = ManuallyDrop::new(self);
                this.ptr.as_ptr()
            }

            /// Reconstructs a `ThinCell<T>` from a raw pointer.
            ///
            /// # Safety
            /// The pointer must have been obtained from a previous call to
            /// [`ThinCell::leak`], and the [`ThinCell`] must not have been dropped in
            /// the meantime.
            pub unsafe fn from_raw(ptr: *mut ()) -> Self {
                ThinCell {
                    // SAFETY: caller guarantees `ptr` is valid
                    ptr: unsafe { NonNull::new_unchecked(ptr) },
                    _marker: PhantomData,
                }
            }

            /// Returns the number of strong owners.
            pub fn count(&self) -> usize {
                self.state().load().count()
            }

            /// Borrows the value mutably.
            ///
            /// Returns a [`Ref`] guard that provides mutable access to the inner value.
            /// The borrow lasts until the guard is dropped. See module-level documentation
            /// for details on borrowing behavior.
            ///
            /// # Examples
            ///
            /// ```
            /// # use thin_cell::unsync::ThinCell;
            /// let cell = ThinCell::new(5);
            ///
            /// {
            ///     let mut borrowed = cell.borrow();
            ///     *borrowed = 10;
            /// } // borrow is released here
            ///
            /// assert_eq!(*cell.borrow(), 10);
            /// ```
            pub fn borrow(&self) -> Ref<'_, T> {
                let inner = self.inner();
                inner.state.borrow();

                // SAFETY: We have exclusive access via borrow flag and block further access
                // with `Ordering::Acquire`/`Release` pair.
                let value = unsafe { &mut *inner.data.get() };

                Ref {
                    value,
                    state: &inner.state,
                }
            }

            /// Attempts to borrow the value mutably.
            ///
            /// Returns `Some(Ref)` if the value is not currently borrowed, or `None` if
            /// it is already borrowed.
            ///
            /// This is the non-blocking variant of [`borrow`](ThinCell::borrow).
            ///
            /// # Examples
            ///
            /// ```
            /// # use thin_cell::unsync::ThinCell;
            /// let cell = ThinCell::new(5);
            ///
            /// let borrow1 = cell.borrow();
            /// assert!(cell.try_borrow().is_none()); // Already borrowed
            /// drop(borrow1);
            /// assert!(cell.try_borrow().is_some()); // Now available
            /// ```
            pub fn try_borrow(&self) -> Option<Ref<'_, T>> {
                let inner = self.inner();
                if !inner.state.try_borrow() {
                    return None;
                }

                // SAFETY: We have exclusive access via borrow flag and block further access
                // with `Ordering::Acquire`/`Release` pair.
                let value = unsafe { &mut *inner.data.get() };

                Some(Ref {
                    value,
                    state: &inner.state,
                })
            }

            /// Get a mutable reference to the inner value without any checks.
            ///
            /// # Safety
            ///
            /// The caller must guarantee that there are no other owners and it is not
            /// borrowed now and during the entire lifetime of the returned reference.
            pub unsafe fn borrow_unchecked(&mut self) -> &mut T {
                let inner = self.inner();
                unsafe { &mut *inner.data.get() }
            }

            /// Creates a new `ThinCell<U>` from `data: U` and coerces it to
            /// `ThinCell<T>`.
            ///
            /// # Safety
            ///
            /// `coerce` function must ensure the returned pointer is:
            ///
            /// - a valid unsizing of `Inner<T>`, e.g., some `Inner<dyn Trait>` or
            ///   `Inner<[_]>`
            /// - with same address (bare data pointer without metadata) as input
            pub unsafe fn new_unsize<U>(
                data: U,
                coerce: impl Fn(*const Inner<U>) -> *const Inner<T>,
            ) -> Self {
                let this = ThinCell::new(data);
                // SAFETY: We're holding unique ownership and is not borrowed.
                unsafe { this.unsize_unchecked(coerce) }
            }

            /// Manually coerce to unsize.
            ///
            /// # Safety
            ///
            /// `coerce` has the same requirements as [`ThinCell::new_unsize`].
            ///
            /// # Panics
            ///
            /// Panics if the `ThinCell` is currently shared (count > 1) or borrowed.
            ///
            /// See [`ThinCell::unsize_unchecked`] for details.
            pub unsafe fn unsize<U: ?Sized>(
                self,
                coerce: impl Fn(*const Inner<T>) -> *const Inner<U>,
            ) -> ThinCell<U> {
                let inner = self.inner();
                let s = inner.state.load();

                assert!(!s.is_shared(), "Cannot coerce shared `ThinCell`");
                assert!(!s.is_borrowed(), "Cannot coerce borrowed `ThinCell`");

                // SAFETY: As tested above, the `ThinCell` is:
                // - not shared, and
                // - not borrowed
                // - validity of `coerce` is guaranteed by caller
                unsafe { self.unsize_unchecked(coerce) }
            }

            /// Manually coerce to unsize without checks.
            ///
            /// # Safety
            ///
            /// - The `ThinCell` must have unique ownership (count == 1)
            /// - The `ThinCell` must not be borrowed
            /// - `coerce` has the same requirements as [`ThinCell::new_unsize`].
            ///
            /// In particular, first two requirement is the exact state after
            /// [`ThinCell::new`].
            pub unsafe fn unsize_unchecked<U: ?Sized>(
                self,
                coerce: impl Fn(*const Inner<T>) -> *const Inner<U>,
            ) -> ThinCell<U> {
                let this = ManuallyDrop::new(self);

                let old_ptr = this.inner_ptr();
                let fat_ptr = coerce(old_ptr);

                let FatPtr { ptr, metadata } = FatPtr::from_ptr::<Inner<U>>(fat_ptr);

                // SAFETY: `Inner` is `repr(C)` and has `metadata` at offset 0
                unsafe { *(old_ptr as *mut usize) = metadata };

                ThinCell {
                    // SAFETY: `ptr` is valid as it comes from `self`
                    ptr: unsafe { NonNull::new_unchecked(ptr) },
                    _marker: PhantomData,
                }
            }

            /// Returns the raw pointer to the inner allocation.
            pub fn as_ptr(&self) -> *const () {
                self.ptr.as_ptr()
            }

            /// Returns `true` if the two `ThinCell`s point to the same allocation.
            pub fn ptr_eq(&self, other: &Self) -> bool {
                std::ptr::eq(self.as_ptr(), other.as_ptr())
            }

            /// Downcasts the `ThinCell<T>` to `ThinCell<U>`.
            ///
            /// # Safety
            ///
            /// The caller must make sure that the inner value is actually of type `U`.
            pub unsafe fn downcast_unchecked<U>(self) -> ThinCell<U> {
                let this = ManuallyDrop::new(self);

                ThinCell {
                    ptr: this.ptr,
                    _marker: PhantomData,
                }
            }
        }

        #[cfg(feature = "weak")]
        impl<T: ?Sized> ThinCell<T> {
            /// Returns the number of strong references.
            pub fn strong_count(&self) -> usize {
                self.state().load().strong_count()
            }

            /// Returns the number of weak references.
            pub fn weak_count(&self) -> usize {
                self.state().load().weak_count()
            }

            /// Creates a new weak reference to this `ThinCell`.
            ///
            /// # Examples
            ///
            /// ```
            /// # #[cfg(feature = "weak")]
            /// # {
            /// # use thin_cell::sync::ThinCell;
            /// let cell = ThinCell::new(42);
            /// let weak = cell.downgrade();
            ///
            /// assert_eq!(cell.strong_count(), 1);
            /// assert_eq!(cell.weak_count(), 2);
            ///
            /// drop(cell);
            /// assert!(weak.upgrade().is_none());
            /// # }
            /// ```
            pub fn downgrade(&self) -> Weak<T> {
                self.state().inc_weak();
                Weak {
                    ptr: self.ptr,
                    _marker: PhantomData,
                }
            }
        }

        impl<T, const N: usize> ThinCell<[T; N]> {
            /// Coerce an array [`ThinCell`] to a slice one.
            pub fn unsize_slice(self) -> ThinCell<[T]> {
                // Safety: unsized coercion from array to slice is safe
                unsafe { self.unsize(|ptr| ptr as _) }
            }
        }

        /// Error returned by [`ThinCell::downcast`] when downcasting fails.
        #[derive(Debug)]
        pub enum DowncastError<T: ?Sized> {
            /// The [`ThinCell`] is currently borrowed.
            Borrowed(ThinCell<T>),

            /// The inner value is not of the target type.
            Type(ThinCell<T>),
        }

        impl<T: ?Sized> DowncastError<T> {
            /// Consumes the error and returns the original `ThinCell<T>`.
            pub fn into_inner(self) -> ThinCell<T> {
                match self {
                    DowncastError::Borrowed(cell) | DowncastError::Type(cell) => cell,
                }
            }
        }

        impl<T: Any + ?Sized> ThinCell<T> {
            /// Attempts to downcast the `ThinCell<T>` to `ThinCell<U>`.
            ///
            /// # Returns
            ///
            /// - `Ok(ThinCell<U>)` if the inner value is of type `U` and is not
            ///   currently borrowed
            /// - `Err(DowncastError::Borrowed(self))` if the inner value is currently
            ///   borrowed
            /// - `Err(DowncastError::Type(self))` if the inner value is not of type `U`
            pub fn downcast<U: Any>(self) -> Result<ThinCell<U>, DowncastError<T>> {
                let inner = self.inner();
                if !inner.state.try_borrow() {
                    return Err(DowncastError::Borrowed(self));
                }

                // SAFETY: We have exclusive access via borrow flag.
                let data_ref = unsafe { &*inner.data.get() };
                let type_id = data_ref.type_id();
                inner.state.unborrow();

                if TypeId::of::<U>() == type_id {
                    // SAFETY: We have verified that the inner value is of type `U`
                    Ok(unsafe { self.downcast_unchecked::<U>() })
                } else {
                    Err(DowncastError::Type(self))
                }
            }
        }

        /// `ThinCell` is `Unpin` as it does not move its inner data.
        impl<T: ?Sized> Unpin for ThinCell<T> {}

        impl<'a, T: ?Sized> Drop for Ref<'a, T> {
            fn drop(&mut self) {
                self.state.unborrow();
            }
        }

        impl<'a, T: ?Sized> Deref for Ref<'a, T> {
            type Target = T;

            fn deref(&self) -> &T {
                self.value
            }
        }

        impl<'a, T: ?Sized> DerefMut for Ref<'a, T> {
            fn deref_mut(&mut self) -> &mut T {
                self.value
            }
        }

        impl<'a, T: Debug + ?Sized> Debug for Ref<'a, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(&**self, f)
            }
        }

        impl<'a, T: Display + ?Sized> Display for Ref<'a, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                Display::fmt(&**self, f)
            }
        }

        impl<T: ?Sized> Clone for ThinCell<T> {
            fn clone(&self) -> Self {
                self.state().inc();

                ThinCell {
                    ptr: self.ptr,
                    _marker: PhantomData,
                }
            }
        }

        #[cfg(not(feature = "weak"))]
        impl<T: ?Sized> Drop for ThinCell<T> {
            fn drop(&mut self) {
                let inner = self.inner();
                if !inner.state.dec() {
                    // Not last owner, nothing to do
                    return;
                }

                // SAFETY: We are the last owner, so we have unique ownership, and we're not
                // using `self` after this call.
                unsafe {
                    self.drop_in_place();
                }
            }
        }

        #[cfg(feature = "weak")]
        impl<T: ?Sized> Drop for ThinCell<T> {
            fn drop(&mut self) {
                let inner = self.inner();
                if !inner.state.dec() {
                    // Not last strong owner, nothing to do
                    return;
                }

                // Keep the strong side's collective weak ref alive as a real `Weak`
                // guard so unwinding through `T::drop` still releases the allocation.
                let _weak: Weak<T> = Weak {
                    ptr: self.ptr,
                    _marker: PhantomData,
                };

                // Drop the value in place.
                // SAFETY: We are the last strong owner, so we have unique ownership.
                unsafe {
                    std::ptr::drop_in_place(inner.data.get());
                }
            }
        }

        #[cfg(feature = "weak")]
        impl<T: ?Sized> Weak<T> {
            /// Reconstructs the raw pointer to the inner allocation.
            fn inner_ptr(&self) -> *const Inner<T> {
                let ptr = self.ptr.as_ptr();

                if ThinCell::<T>::IS_SIZED {
                    // SIZED CASE: Cast pointer-to-pointer
                    let ptr_ref = &ptr as *const *mut () as *const *const Inner<T>;
                    unsafe { *ptr_ref }
                } else {
                    // UNSIZED CASE: Read metadata
                    let metadata = unsafe { *(ptr as *const usize) };
                    FatPtr { ptr, metadata }.into_ptr()
                }
            }

            /// Returns a reference to the inner allocation.
            fn inner(&self) -> &Inner<T> {
                unsafe { &*self.inner_ptr() }
            }

            /// Returns a reference to the state cell.
            fn state(&self) -> &State {
                &self.inner().state
            }

            /// Attempts to upgrade the weak reference to a strong reference.
            ///
            /// Returns `Some(ThinCell)` if the value still exists, or `None` if it
            /// has been dropped.
            ///
            /// # Examples
            ///
            /// ```
            /// # #[cfg(feature = "weak")]
            /// # {
            /// # use thin_cell::sync::ThinCell;
            /// let cell = ThinCell::new(42);
            /// let weak = cell.downgrade();
            ///
            /// let strong = weak.upgrade().unwrap();
            /// assert_eq!(*strong.borrow(), 42);
            ///
            /// drop(cell);
            /// drop(strong);
            /// assert!(weak.upgrade().is_none());
            /// # }
            /// ```
            pub fn upgrade(&self) -> Option<ThinCell<T>> {
                let state = self.state();

                // Atomically try to increment strong count only if non-zero
                if !state.try_inc() {
                    return None;
                }

                Some(ThinCell {
                    ptr: self.ptr,
                    _marker: PhantomData,
                })
            }

            /// Returns the number of strong references.
            pub fn strong_count(&self) -> usize {
                self.state().load().strong_count()
            }

            /// Returns the number of weak references.
            pub fn weak_count(&self) -> usize {
                self.state().load().weak_count()
            }

            /// Gets a raw pointer to the inner allocation.
            ///
            /// The pointer is valid only if the strong count is non-zero.
            pub fn as_ptr(&self) -> *const () {
                self.ptr.as_ptr()
            }
        }

        #[cfg(feature = "weak")]
        impl<T: ?Sized> Clone for Weak<T> {
            fn clone(&self) -> Self {
                self.state().inc_weak();
                Weak {
                    ptr: self.ptr,
                    _marker: PhantomData,
                }
            }
        }

        #[cfg(feature = "weak")]
        impl<T: ?Sized> Drop for Weak<T> {
            fn drop(&mut self) {
                let inner = self.inner();
                if !inner.state.dec_weak() {
                    // Not last weak ref, nothing to do
                    return;
                }

                // Last weak ref and no strong refs - deallocate memory only
                // The value T was already dropped when the last strong ref was dropped
                // SAFETY: We are the last weak owner and no strong owners exist
                unsafe {
                    let layout = std::alloc::Layout::for_value(inner);
                    std::alloc::dealloc(self.inner_ptr() as *mut u8, layout);
                }
            }
        }

        #[cfg(feature = "weak")]
        impl<T: ?Sized> Debug for Weak<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "(Weak)")
            }
        }

        impl<T: Default> Default for ThinCell<T> {
            fn default() -> Self {
                ThinCell::new(T::default())
            }
        }

        impl<T: Debug + ?Sized> Debug for ThinCell<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let inner = self.inner();
                let state = inner.state.load();
                let mut d = f.debug_struct("ThinCell");
                match self.try_borrow() {
                    Some(borrowed) => d.field("value", &borrowed),
                    None => d.field("value", &"<borrowed>"),
                }
                .field("state", &state)
                .finish()
            }
        }

        impl<T: Display + ?Sized> Display for ThinCell<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.try_borrow() {
                    Some(borrowed) => Display::fmt(&*borrowed, f),
                    None => write!(f, "<borrowed>"),
                }
            }
        }

        impl<T: PartialEq + ?Sized> PartialEq<ThinCell<T>> for ThinCell<T> {
            /// Compares the inner values for equality.
            ///
            /// This will block on `sync` version or panic on `unsync` version if either `ThinCell` is currently borrowed.
            ///
            /// If a shallow comparison is desired, use [`ptr_eq`](ThinCell::ptr_eq)
            /// instead.
            fn eq(&self, other: &Self) -> bool {
                self.borrow().eq(&other.borrow())
            }
        }

        impl<T: Eq + ?Sized> Eq for ThinCell<T> {}

        #[allow(clippy::non_canonical_partial_ord_impl)]
        impl<T: Ord + ?Sized> PartialOrd<ThinCell<T>> for ThinCell<T> {
            /// Compares the inner values.
            ///
            /// This will block on `sync` version or panic on `unsync` version if either `ThinCell` is currently borrowed.
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.borrow().partial_cmp(&other.borrow())
            }
        }

        impl<T: Ord + ?Sized> Ord for ThinCell<T> {
            /// Compares the inner values.
            ///
            /// This will block on `sync` version or panic on `unsync` version if either `ThinCell` is currently borrowed.
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.borrow().cmp(&other.borrow())
            }
        }
    }
}

use thin_cell;
