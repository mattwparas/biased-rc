use std::alloc::Layout;
use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::Infallible;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::ptr::NonNull;
use std::{cell::Cell, sync::atomic::Ordering, thread::current};

use atomic::Atomic;
use bytemuck::NoUninit;

use core::convert::TryInto;
use core::num::NonZeroUsize;
use core::sync::atomic::AtomicUsize;

use std::{alloc, cmp, fmt, mem, ptr};

use std::hash::{Hash, Hasher};

thread_local! {
    /// Zero-sized thread-local variable to differentiate threads.
    static THREAD_MARKER: () = ();
}

const SENITEL: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(usize::MAX) };

/// A unique identifier for a running thread.
///
/// Uniqueness is guaranteed between running threads. However, the ids of dead
/// threads may be reused.
///
/// There is a chance that this implementation can be replaced by [`std::thread::ThreadId`]
/// when [`as_u64()`] is stabilized.
///
/// **Note:** The current (non platform specific) implementation uses the address of a
/// thread local static variable for thread identification.
///
/// [`as_u64()`]: std::thread::ThreadId::as_u64
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub(crate) struct ThreadId(pub(crate) NonZeroUsize);

impl ThreadId {
    /// Creates a new `ThreadId` for the given raw id.
    #[inline(always)]
    pub(crate) const fn new(value: NonZeroUsize) -> Self {
        Self(value)
    }

    /// Gets the id for the thread that invokes it.
    #[inline]
    pub(crate) fn current_thread() -> Self {
        Self::new(
            THREAD_MARKER
                .try_with(|x| x as *const _ as usize)
                .expect("the thread's local data has already been destroyed")
                .try_into()
                .expect("thread id should never be zero"),
        )
    }
}

impl PartialEq for ThreadId {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (SENITEL, _) | (_, SENITEL) => false,
            (a, b) => a == b,
        }
    }
}

/// An [`Option`]`<`[`ThreadId`]`>` which can be safely shared between threads.
///
/// **Note:** Currently implemented as a wrapper around [`AtomicUsize`].
#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct AtomicOptionThreadId(core::sync::atomic::AtomicUsize);

/// Converts the internal representation in [`AtomicOptionThreadId`] into
/// `Some(ThreadId)` or `None`.
///
/// This should only be a type-level conversion and a noop at runtime.
#[inline(always)]
fn wrap(value: usize) -> Option<ThreadId> {
    match value {
        0 => None,
        n => Some(ThreadId::new(n.try_into().unwrap())),
    }
}

/// Converts an `Option<ThreadId>` into the internal representation for
/// [`AtomicOptionThreadId`].
///
/// This should only be a type-level conversion and a noop at runtime.
#[inline(always)]
const fn unwrap(value: Option<ThreadId>) -> usize {
    match value {
        None => 0,
        Some(id) => id.0.get(),
    }
}

impl AtomicOptionThreadId {
    /// Creates a new `AtomicOptionThreadId`.
    #[inline]
    pub const fn new(id: Option<ThreadId>) -> Self {
        Self(AtomicUsize::new(unwrap(id)))
    }

    /// Loads a value from the atomic.
    ///
    /// The [`Ordering`] may only be `SeqCst`, `Acquire` or `Relaxed`.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Release` or `AcqRel`.
    #[inline]
    pub fn load(&self, order: Ordering) -> Option<ThreadId> {
        wrap(self.0.load(order))
    }

    /// Stores a value into the atomic.
    ///
    /// The [`Ordering`] may only be `SeqCst`, `Release` or `Relaxed`.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Acquire` or `AcqRel`.
    #[inline]
    pub fn store(&self, val: Option<ThreadId>, order: Ordering) {
        self.0.store(unwrap(val), order);
    }

    /// Stores `new` into the atomic iff the currently stored value is `None`.
    ///
    /// The success ordering may be any [`Ordering`], but `failure` may only be
    /// `SeqCst`, `Release` or `Relaxed` and must be equivalent to or weaker than
    /// `success`.
    ///
    /// # Panics
    ///
    /// Panics if `failure` is `Acquire`, `AcqRel` or a stronger ordering than
    /// `success`.
    #[inline]
    pub fn store_if_none(
        &self,
        new: Option<ThreadId>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<ThreadId>, Option<ThreadId>> {
        self.0
            .compare_exchange(unwrap(None), unwrap(new), success, failure)
            .map(wrap)
            .map_err(wrap)
    }
}

impl Default for AtomicOptionThreadId {
    /// Creates an `AtomicOptionThreadId` initialized to [`None`].
    #[inline]
    fn default() -> Self {
        Self::new(None)
    }
}

impl From<ThreadId> for AtomicOptionThreadId {
    /// Converts a [`ThreadId`] into an `AtomicOptionThreadId`, wrapping it in [`Some`].
    #[inline]
    fn from(id: ThreadId) -> Self {
        Self::new(Some(id))
    }
}

impl From<Option<ThreadId>> for AtomicOptionThreadId {
    /// Converts an [`Option`]`<`[`ThreadId`]`>` into an `AtomicOptionThreadId`.
    #[inline]
    fn from(id: Option<ThreadId>) -> Self {
        Self::new(id)
    }
}

pub struct RcWord {
    thread_id: Cell<Option<ThreadId>>,
    biased_counter: Cell<usize>,
    shared: Atomic<Shared>,
}

#[repr(C, packed)]
#[derive(Copy, Clone, PartialEq, Eq, NoUninit, Debug)]
pub struct Shared {
    counter: u32,
    merged: bool,
    queued: bool,
    _align: [i8; 2],
}

#[repr(C)]
pub struct RcBox<T: ?Sized> {
    rcword: RcWord,
    data: T,
}

impl RcWord {
    pub fn new() -> Self {
        Self {
            thread_id: Cell::new(Some(ThreadId::current_thread())),
            biased_counter: Cell::new(1),
            shared: Atomic::new(Shared {
                counter: 0,
                merged: false,
                queued: false,
                _align: Default::default(),
            }),
        }
    }
}

impl<T: ?Sized> RcBox<T> {
    // TODO: Lift this to the Obj struct that eventually gets made
    pub fn increment(&self) {
        let owner_tid = self.rcword.thread_id.get();
        let my_tid = ThreadId::current_thread();

        if owner_tid == Some(my_tid) {
            self.fast_increment();
        } else {
            self.slow_increment();
        }
    }

    pub fn fast_increment(&self) {
        let counter = self.rcword.biased_counter.get();
        if counter == usize::MAX {
            panic!("reference counter overflow");
        }
        self.rcword.biased_counter.set(counter + 1);
    }

    pub fn slow_increment(&self) {
        loop {
            // TODO: Do some reading on the memory implications here
            // Do we have to read the whole thing together?
            let old = self.rcword.shared.load(Ordering::Relaxed);
            let mut new = old;
            new.counter += 1;

            if self
                .rcword
                .shared
                .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    pub fn decrement(&self) -> bool {
        let owner_tid = self.rcword.thread_id.get();
        let my_tid = ThreadId::current_thread();

        if owner_tid == Some(my_tid) {
            self.fast_decrement()
        } else {
            self.slow_decrement()
        }
    }

    pub fn fast_decrement(&self) -> bool {
        self.rcword.biased_counter.update(|x| x - 1);
        if self.rcword.biased_counter.get() > 0 {
            return false;
        }

        let mut new;

        loop {
            let old = self.rcword.shared.load(Ordering::Relaxed);
            new = old;
            new.merged = true;
            if self
                .rcword
                .shared
                .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        if new.counter == 0 {
            // self.deallocate()

            true
        } else {
            self.rcword.thread_id.set(None);

            false
        }
    }

    pub fn slow_decrement(&self) -> bool {
        let mut old;
        let mut new;
        loop {
            old = self.rcword.shared.load(Ordering::Relaxed);
            new = old;

            if new.counter == 0 {
                new.queued = true;
            } else {
                new.counter -= 1;
            }

            if self
                .rcword
                .shared
                .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        if old.queued != new.queued {
            self.queue();
            false
        } else if new.merged && new.counter == 0 {
            // self.deallocate()
            true
        } else {
            false
        }
    }

    pub fn queue(&self) {
        // TODO:
        todo!()
    }
}

// Set up the queue to save ourselves... or something
struct QueuedObjects {
    // TODO: Except don't have this just be TypeId -> T,
    // just have it be like, HashMap<T, Vec<HybridRc<dyn ANy>>>
    map: HashMap<TypeId, Vec<Box<dyn Any>>>,
}

struct TypeMap {
    layouts: HashMap<TypeId, Layout>,
    inner: HashMap<TypeId, Box<dyn Any>>,
}

impl TypeMap {
    fn insert<T: Any + 'static>(&mut self, value: T) {
        let layout = Layout::for_value(&value);

        // Get the layout for the type on the way out?
        self.layouts.insert(TypeId::of::<T>(), layout);
        self.inner.insert(TypeId::of::<T>(), Box::new(value));
    }

    fn get<T: Any + 'static>(&self) -> Option<&T> {
        self.inner
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }
}

impl<T: ?Sized> RcBox<T> {
    /// Deallocates an `RcBox`
    ///
    /// `meta` will be dropped, but `data` must have already been dropped in place.
    ///
    /// # Safety
    /// The allocation must have been previously allocated with `RcBox::allocate_*()`.
    #[inline]
    unsafe fn dealloc(ptr: NonNull<RcBox<T>>) {
        unsafe { ptr::addr_of_mut!((*ptr.as_ptr()).rcword).drop_in_place() };
        let layout = Layout::for_value(unsafe { ptr.as_ref() });
        unsafe { alloc::dealloc(ptr.as_ptr().cast(), layout) };
    }

    /// Tries to allocate an `RcBox` for a possibly dynamically sized value
    ///
    /// Size and alignment of `example` are used for allocation and if `example` is a fat reference
    /// the pointer metadata is copied to the resulting pointer.
    ///
    /// Returns a mutable pointer on success and the memory layout that could not be allocated
    /// if the allocation failed.
    #[inline]
    fn try_allocate_for_val(
        meta: RcWord,
        example: &T,
        zeroed: bool,
    ) -> Result<NonNull<RcBox<T>>, Layout> {
        let layout = Layout::new::<RcBox<()>>();
        let layout = layout
            .extend(Layout::for_value(example))
            .map_err(|_| layout)?
            .0
            .pad_to_align();

        // Allocate memory
        let ptr = unsafe {
            if zeroed {
                alloc::alloc_zeroed(layout)
            } else {
                alloc::alloc(layout)
            }
        }
        .cast::<RcBox<()>>();

        // Write RcMeta fields
        // Safety: Freshly allocated, so valid to write to.
        unsafe { ptr::addr_of_mut!((*ptr).rcword).write(meta) };

        // Combine metadata from `example` with new memory
        let result = set_ptr_value(example, ptr);

        NonNull::new(result as *mut RcBox<T>).ok_or(layout)
    }

    /// Allocates an `RcBox` for a possibly dynamically sized value
    ///
    /// Size and alignment of `example` are used for allocation and if `example` is a fat reference
    /// the pointer metadata is copied to the resulting pointer.
    ///
    /// Returns a mutable pointer on success.
    ///
    /// # Panics
    /// Panics or aborts if the allocation failed.
    #[inline]
    fn allocate_for_val(meta: RcWord, example: &T, zeroed: bool) -> NonNull<RcBox<T>> {
        match Self::try_allocate_for_val(meta, example, zeroed) {
            Ok(result) => result,
            Err(layout) => alloc::handle_alloc_error(layout),
        }
    }

    /// Get the pointer to a `RcBox<T>` from a pointer to the data
    ///
    /// # Safety
    ///
    /// The pointer must point to (and have valid metadata for) the data part of a previously
    /// valid instance of `RcBox<T>` and it must not be dangling.
    #[inline]
    unsafe fn ptr_from_data_ptr(ptr: *const T) -> *const RcBox<T> {
        // Calculate layout of RcBox<T> without `data` tail, but including padding
        let base_layout = Layout::new::<RcBox<()>>();
        // Safety: covered by the safety contract above
        let value_alignment = mem::align_of_val(unsafe { &*ptr });
        let value_offset_layout =
            Layout::from_size_align(0, value_alignment).expect("invalid memory layout");
        let layout = base_layout
            .extend(value_offset_layout)
            .expect("invalid memory layout")
            .0;

        // Move pointer to point to the start of the original RcBox<T>
        // Safety: covered by the safety contract above
        let rcbox = unsafe { ptr.cast::<u8>().offset(-(layout.size() as isize)) };
        set_ptr_value(ptr, rcbox as *mut u8) as *const RcBox<T>
    }
}

impl<T> RcBox<T> {
    /// Tries to allocate an `RcBox`
    ///
    /// Returns a mutable reference with arbitrary lifetime on success and the memory layout that
    /// could not be allocated if the allocation failed.
    #[inline]
    fn try_allocate(meta: RcWord) -> Result<NonNull<RcBox<mem::MaybeUninit<T>>>, Layout> {
        let layout = Layout::new::<RcBox<T>>();

        let ptr = unsafe { alloc::alloc(layout) }.cast::<RcBox<mem::MaybeUninit<T>>>();
        if ptr.is_null() {
            Err(layout)
        } else {
            unsafe { ptr::addr_of_mut!((*ptr).rcword).write(meta) };
            Ok(unsafe { NonNull::new_unchecked(ptr) })
        }
    }

    /// Allocates an `RcBox`
    ///
    /// Returns a mutable reference with arbitrary lifetime on success.
    ///
    /// # Panics
    /// Panics or aborts if the allocation failed.
    #[inline]
    fn allocate(meta: RcWord) -> NonNull<RcBox<mem::MaybeUninit<T>>> {
        match Self::try_allocate(meta) {
            Ok(result) => result,
            Err(layout) => alloc::handle_alloc_error(layout),
        }
    }

    /// Tries to allocate an `RcBox` for a slice.
    ///
    /// Returns a mutable reference with arbitrary lifetime on success and the memory layout that
    /// could not be allocated if the allocation failed or the layout calculation overflowed.
    #[inline]
    fn try_allocate_slice<'a>(
        meta: RcWord,
        len: usize,
        zeroed: bool,
    ) -> Result<&'a mut RcBox<[mem::MaybeUninit<T>]>, Layout> {
        // Calculate memory layout
        let layout = Layout::new::<RcBox<[T; 0]>>();
        let payload_layout = Layout::array::<T>(len).map_err(|_| layout)?;
        let layout = layout
            .extend(payload_layout)
            .map_err(|_| layout)?
            .0
            .pad_to_align();

        // Allocate memory
        let ptr = unsafe {
            if zeroed {
                alloc::alloc_zeroed(layout)
            } else {
                alloc::alloc(layout)
            }
        };

        // Build a fat pointer
        // The immediate slice reference [MaybeUninit<u8>] *should* be sound
        let ptr = ptr::slice_from_raw_parts_mut(ptr.cast::<mem::MaybeUninit<u8>>(), len)
            as *mut RcBox<[mem::MaybeUninit<T>]>;

        if ptr.is_null() {
            // Allocation failed
            Err(layout)
        } else {
            // Initialize metadata field and return result
            unsafe { ptr::addr_of_mut!((*ptr).rcword).write(meta) };
            Ok(unsafe { ptr.as_mut().unwrap() })
        }
    }

    /// Allocates an `RcBox` for a slice
    ///
    /// Returns a mutable reference with arbitrary lifetime on success.
    ///
    /// # Panics
    /// Panics or aborts if the allocation failed or the memory layout calculation overflowed.
    #[inline]
    fn allocate_slice<'a>(
        meta: RcWord,
        len: usize,
        zeroed: bool,
    ) -> &'a mut RcBox<[mem::MaybeUninit<T>]> {
        match Self::try_allocate_slice(meta, len, zeroed) {
            Ok(result) => result,
            Err(layout) => alloc::handle_alloc_error(layout),
        }
    }
}

impl<T> RcBox<mem::MaybeUninit<T>> {
    /// Converts to a mutable reference without the `MaybeUninit` wrapper.
    ///
    /// # Safety
    /// The payload must have been fully initialized or this causes immediate undefined behaviour.
    #[inline]
    unsafe fn assume_init(&mut self) -> &mut RcBox<T> {
        unsafe { (self as *mut Self).cast::<RcBox<T>>().as_mut() }.unwrap()
    }
}

impl<T> RcBox<[mem::MaybeUninit<T>]> {
    /// Converts to a mutable reference without the `MaybeUninit` wrapper.
    ///
    /// # Safety
    /// The payload slice must have been fully initialized or this causes immediate undefined
    /// behaviour.
    #[inline]
    unsafe fn assume_init(&mut self) -> &mut RcBox<[T]> {
        unsafe { (self as *mut _ as *mut RcBox<[T]>).as_mut() }.unwrap()
    }
}

/// Reimplementation of `ptr::set_ptr_value` as long as that one is unstable
///
/// Constructs a new pointer to `addr_ptr` with the metadata and type of `meta_ptr`.
#[inline]
fn set_ptr_value<T: ?Sized, U>(mut meta_ptr: *const T, addr_ptr: *mut U) -> *mut T {
    let thin = (&mut meta_ptr as *mut *const T).cast::<*const u8>();
    // Safety: In case of a thin pointer, this operations is identical
    // to a simple assignment. In case of a fat pointer, with the current
    // fat pointer layout implementation, the first field of such a
    // pointer is always the data pointer, which is likewise assigned.
    unsafe { *thin = addr_ptr.cast() };

    meta_ptr as *mut T
}

pub struct HybridRc<T: ?Sized> {
    ptr: NonNull<RcBox<T>>,
    phantom2: PhantomData<RcBox<T>>,
}

impl<T: ?Sized> HybridRc<T> {
    /// Creates a new `HybridRc` from a pointer to a shared allocation.
    ///
    /// The reference counters must have been updated by the caller.
    #[inline(always)]
    fn from_inner(ptr: NonNull<RcBox<T>>) -> Self {
        Self {
            ptr,
            phantom2: PhantomData,
        }
    }

    #[inline(always)]
    fn get_box(&self) -> &RcBox<T> {
        unsafe { &(*self.ptr.as_ptr()) }
    }

    /// Provides a reference to the inner value.
    #[inline(always)]
    fn data(&self) -> &T {
        // Safety: as long as one HybridRc or Weak for this item exists, the memory stays allocated.
        unsafe { &(*self.ptr.as_ptr()).data }
    }

    /// Provides a reference to the shared metadata.
    #[inline(always)]
    fn meta(&self) -> &RcWord {
        // todo!()
        // Safety: as long as one HybridRc or Weak for this item exists, the memory stays allocated.
        unsafe { &(*self.ptr.as_ptr()).rcword }
    }

    /// Provides a reference to the inner `HybridRc` of a `Pin<HybridRc<T>>`
    ///
    /// # Safety
    /// The caller must ensure that the reference is not used to move the value out of self.
    #[inline(always)]
    unsafe fn pin_get_ref(this: &Pin<Self>) -> &Self {
        // SAFETY: Pin is repr(transparent) and by contract the caller doesn't use the reference
        // to move the value.
        unsafe { &*(this as *const Pin<Self>).cast::<Self>() }
    }

    #[must_use]
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        unsafe { &mut (*this.ptr.as_ptr()).data }
    }

    /// Returns a mutable reference to the value, iff the value is not shared
    /// with another `HybridRc` or [`Weak`].
    ///
    /// Returns `None` otherwise.
    #[must_use]
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        todo!()

        // if this.meta().has_unique_ref(!State::SHARED) {
        //     unsafe { Some(Self::get_mut_unchecked(this)) }
        // } else {
        //     None
        // }
    }

    /// Provides a raw pointer to the referenced value
    ///
    /// The counts are not affected in any way and the `HybridRc` is not consumed.
    /// The pointer is valid for as long there exists at least one `HybridRc` for the value.
    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr = this.ptr.as_ptr();

        // Safety: Neccessary for `from_raw()` (when implemented), retains provenance.
        // Besides that, does basically the same thing as `data()` or `get_mut_unchecked()`.
        unsafe { ptr::addr_of_mut!((*ptr).data) }
    }

    /// Consumes the `HybridRc<T, State>`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to a `HybridRc` using
    /// [`HybridRc<T, State>::from_raw()`].
    #[must_use = "Memory will leak if the result is not used"]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = Self::as_ptr(&this);
        mem::forget(this);
        ptr
    }

    /// Reconstructs a `HybridRc<T, State>` from a raw pointer.
    ///
    /// Creates a `HybridRc<T, State>` from a pointer that has been previously returned by
    /// a call to [`into_raw()`].
    ///
    /// # Safety
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`HybridRc<T, State>`][`into_raw()`] for the same `State` *and* the same `T` or another
    /// compatible type that has the same size and alignment. The latter case amounts to
    /// [`mem::transmute()`] and is likely to produce undefined behaviour if not handled correctly.
    ///
    /// The value must not have been dropped yet.
    ///
    /// [`into_raw()`]: Self::into_raw
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // Safety: covered by the safety contract for this function
        let box_ptr = unsafe { RcBox::<T>::ptr_from_data_ptr(ptr) };

        Self::from_inner(NonNull::new(box_ptr as *mut _).expect("invalid pointer"))
    }

    /// Checks if two `HybridRc`s point to the same allocation.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &HybridRc<T>) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }

    /// Checks if two pinned `HybridRc`s point to the same allocation.
    #[inline]
    pub fn ptr_eq_pin(this: &Pin<Self>, other: &Pin<HybridRc<T>>) -> bool {
        // SAFETY: we are not moving anything and we don't expose any pointers.
        let this = unsafe { Self::pin_get_ref(this) };
        let other = unsafe { HybridRc::<T>::pin_get_ref(other) };
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        // let meta = this.meta();
        // meta.strong_shared.load(Ordering::SeqCst)
        //     + if State::SHARED {
        //         0
        //     } else {
        //         meta.strong_local.get() - 1
        //     }

        todo!()
    }

    /// Gets the approximate number of strong pointers to the pinned inner value.
    ///
    #[inline]
    pub fn strong_count_pin(this: &Pin<Self>) -> usize {
        // SAFETY: We are not moving anything and we don't expose any pointers.
        let this = unsafe { Self::pin_get_ref(this) };
        Self::strong_count(this)
    }

    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        // match this.meta().weak.load(Ordering::SeqCst) {
        //     // Lock value => there were zero weak references apart from the implicit one.
        //     usize::MAX => 0,
        //     count => count - 1,
        // }

        todo!()
    }

    /// Gets the number of [`PinWeak`] pointers to the pinned inner value.
    ///
    #[inline]
    pub fn weak_count_pin(this: &Pin<Self>) -> usize {
        // SAFETY: We are not moving anything and we don't expose any pointers.
        let this = unsafe { Self::pin_get_ref(this) };
        Self::weak_count(this)
    }

    #[inline]
    fn build_new_meta() -> RcWord {
        RcWord::new()
    }

    /// Drops the contained value and also drops the shared `RcBox` if there are no other `Weak`
    /// references.
    ///
    /// # Safety
    /// Only safe to use in `drop()` or a consuming function after verifying that no other strong
    /// reference exists. Otherwise after calling this e.g. dereferencing the `HybridRc` WILL
    /// cause undefined behaviour and even dropping it MAY cause undefined behaviour.
    unsafe fn drop_contents_and_maybe_box(&mut self) {
        // Safety: only called if this was the last strong reference
        unsafe {
            ptr::drop_in_place(Self::get_mut_unchecked(self));
        }

        // Safety: only called if this was the last (weak) reference
        unsafe {
            RcBox::dealloc(self.ptr);
        }

        // todo!()
    }
}

impl<T> HybridRc<T> {
    #[inline]
    pub fn new(data: T) -> Self {
        let mut inner = RcBox::allocate(Self::build_new_meta());
        let inner = unsafe { inner.as_mut() };
        inner.data.write(data);
        Self::from_inner(unsafe { inner.assume_init() }.into())
    }

    /// Creates a new `HybridRc` with uninitialized contents.
    #[inline]
    pub fn new_uninit() -> HybridRc<mem::MaybeUninit<T>> {
        let inner = RcBox::allocate(Self::build_new_meta());
        HybridRc::from_inner(inner)
    }

    /// Creates a new `HybridRc` with uninitialized contents, with the memory being filled with
    /// 0 bytes.
    ///
    /// See [`MaybeUninit::zeroed()`] for examples of correct and incorrect usage of this method.
    ///
    /// [`MaybeUninit::zeroed()`]: mem::MaybeUninit::zeroed
    #[inline]
    pub fn new_zeroed() -> HybridRc<mem::MaybeUninit<T>> {
        let mut inner = RcBox::allocate(Self::build_new_meta());
        unsafe { inner.as_mut() }.data = mem::MaybeUninit::zeroed();
        HybridRc::from_inner(inner)
    }

    /// Creates a new `Pin<HybridRc<T>>`. If `T` does not implement `Unpin`, then `data` will be
    /// pinned in memory and unable to be moved.
    #[inline]
    pub fn pin(data: T) -> Pin<Self> {
        unsafe { Pin::new_unchecked(Self::new(data)) }
    }

    pub fn try_new(data: T) -> Result<Self, AllocError> {
        let mut inner = RcBox::try_allocate(Self::build_new_meta()).map_err(|_| AllocError)?;
        let inner = unsafe { inner.as_mut() };
        inner.data.write(data);
        Ok(Self::from_inner(unsafe { inner.assume_init() }.into()))
    }

    pub fn try_new_uninit() -> Result<HybridRc<mem::MaybeUninit<T>>, AllocError> {
        let inner = RcBox::try_allocate(Self::build_new_meta()).map_err(|_| AllocError)?;
        Ok(HybridRc::from_inner(inner.into()))
    }

    pub fn try_new_zeroed() -> Result<HybridRc<mem::MaybeUninit<T>>, AllocError> {
        let mut inner = RcBox::try_allocate(Self::build_new_meta()).map_err(|_| AllocError)?;
        unsafe { inner.as_mut() }.data = mem::MaybeUninit::zeroed();
        Ok(HybridRc::from_inner(inner))
    }

    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // if State::SHARED {
        //     Self::try_unwrap_internal(this)
        // } else {
        //     // If we may access the local counter, first check and decrement that one.
        //     let local_count = this.meta().strong_local.get();
        //     if local_count == 1 {
        //         this.meta().strong_local.set(0);
        //         match Self::try_unwrap_internal(this) {
        //             Ok(result) => Ok(result),
        //             Err(this) => {
        //                 this.meta().strong_local.set(local_count);
        //                 Err(this)
        //             }
        //         }
        //     } else {
        //         Err(this)
        //     }
        // }

        todo!()
    }

    /// Returns the inner value, if this `HybridRc` is the only strong reference to it, assuming
    /// that there are no (other) local references to the value.
    ///
    /// Used internally by `try_unwrap()`.
    #[inline]
    fn try_unwrap_internal(this: Self) -> Result<T, Self> {
        // let meta = this.meta();
        // // There is one implicit shared reference for all local references, so if there are no other
        // // local references or we are a shared shared and the shared counter is 1, we are the only
        // // strong reference left.
        // if meta
        //     .strong_shared
        //     .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Relaxed)
        //     .is_err()
        // {
        //     Err(this)
        // } else {
        //     // Relaxed should be enough, as `strong_shared` already hit 0, so no more
        //     // Weak upgrading is possible.
        //     meta.owner.store(None, Ordering::Relaxed);

        //     let copy = unsafe { ptr::read(Self::as_ptr(&this)) };

        //     // Make a weak pointer to clean up the remaining implicit weak reference
        //     let _weak = Weak { ptr: this.ptr };
        //     mem::forget(this);

        //     Ok(copy)
        // }

        todo!()
    }
}

/// The `AllocError` error indicates an allocation failure when using `try_new()` etc.
///
/// Will become a type alias for [`std::alloc::AllocError`] once that is stabilized.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

impl std::error::Error for AllocError {}

impl From<Infallible> for AllocError {
    fn from(_: Infallible) -> AllocError {
        unreachable!();
    }
}

impl<T: ?Sized> Deref for HybridRc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.data()
    }
}

impl<T: ?Sized> Borrow<T> for HybridRc<T> {
    #[inline]
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> AsRef<T> for HybridRc<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> Clone for HybridRc<T> {
    #[inline]
    fn clone(&self) -> Self {
        self.get_box().increment();
        Self::from_inner(self.ptr)
    }
}

impl<T: ?Sized> Drop for HybridRc<T> {
    /// Drops the `HybridRc`.
    ///
    /// This will decrement the appropriate reference count depending on `State`. If both strong
    /// reference counts reach zero then the only other references (if any) are [`Weak`]. In that
    /// case the inner value is dropped.
    #[inline]
    fn drop(&mut self) {
        if self.get_box().decrement() {
            unsafe { self.drop_contents_and_maybe_box() };
        }
    }
}

// Propagate some useful traits implemented by the inner type

impl<T: Default> Default for HybridRc<T> {
    /// Creates a new `HybridRc`, with the `Default` value for `T`.
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T: ?Sized + PartialEq> PartialEq<HybridRc<T>> for HybridRc<T> {
    /// Equality for `HybridRc`s.
    ///
    /// Two `HybridRc`s are equal if their inner values are equal, independent of if they are
    /// stored in the same allocation.
    #[inline]
    fn eq(&self, other: &HybridRc<T>) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq> Eq for HybridRc<T> {}

impl<T: ?Sized + Hash> Hash for HybridRc<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Self::data(self).hash(state);
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd<HybridRc<T>> for HybridRc<T> {
    /// Partial comparison for `HybridRc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    #[inline]
    fn partial_cmp(&self, other: &HybridRc<T>) -> Option<cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for HybridRc<T> {
    /// Comparison for `HybridRc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for HybridRc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&Self::data(self), f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for HybridRc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&Self::data(self), f)
    }
}

// `HybridRc` can be formatted as a pointer.
impl<T: ?Sized> fmt::Pointer for HybridRc<T> {
    /// Formats the value using the given formatter.
    ///
    /// If the `#` flag is used, the state (shared/local) is written after the address.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // if f.alternate() {
        //     fmt::Pointer::fmt(&Self::as_ptr(self), f)?;
        //     f.write_str(if State::SHARED {
        //         " [shared]"
        //     } else {
        //         " [local]"
        //     })
        // } else {
        //     fmt::Pointer::fmt(&Self::as_ptr(self), f)
        // }

        todo!()
    }
}

/// `HybridRc<T>` is always `Unpin` itself, because the data value is on the heap,
/// so moving `HybridRc<T>` doesn't move the content even if `T` is not `Unpin`.
///
/// This allows unpinning e.g. `Pin<Box<HybridRc<T>>>` but not any `Pin<HybridRc<T>>`!
impl<T: ?Sized> Unpin for HybridRc<T> {}

#[test]
fn does_this_work() {
    let value = HybridRc::new(10);
    println!("{}", value);
}

#[test]
fn test_drop_impl() {
    struct Foo {
        foo: usize,
    }
    impl Drop for Foo {
        fn drop(&mut self) {
            println!("Calling drop: {}", self.foo);
        }
    }
    let value = HybridRc::new(Foo { foo: 10 });

    drop(value);
}

#[test]
fn test_clone_impl() {
    struct Foo {
        foo: usize,
    }
    impl Drop for Foo {
        fn drop(&mut self) {
            println!("Calling drop: {}", self.foo);
        }
    }
    let value = HybridRc::new(Foo { foo: 10 });
    let cloned = HybridRc::clone(&value);

    drop(value);

    println!("Now we're done");

    drop(cloned);
}
