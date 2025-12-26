use std::alloc::Layout;
use std::any::Any;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::Infallible;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::AtomicU32;
use std::sync::{LazyLock, Mutex};
use std::{cell::Cell, sync::atomic::Ordering};

use atomic::Atomic;
use bytemuck::NoUninit;

use core::convert::TryInto;
use core::num::NonZeroUsize;

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
#[derive(Debug, Clone, Copy, Hash, Eq)]
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

// Okay, now that this appears to be working, we
// need to shrink this down as much as possible.
pub struct RcWord {
    thread_id: Cell<Option<ThreadId>>,
    biased_counter: Cell<u32>,
    // shared: Atomic<Shared>,
    shared: SharedPacked,
}

#[repr(C, packed)]
#[derive(Copy, Clone, PartialEq, Eq, NoUninit, Debug)]
pub struct Shared {
    counter: i32,
    merged: bool,
    queued: bool,
    _align: [i8; 2],
}

impl Shared {
    fn new() -> Self {
        Self {
            counter: 0,
            merged: false,
            queued: false,
            _align: Default::default(),
        }
    }

    #[inline(always)]
    fn set_queued(&mut self, queued: bool) {
        self.queued = queued;
    }

    #[inline(always)]
    fn set_merged(&mut self, merged: bool) {
        self.merged = merged;
    }

    fn get_queued(&self) -> bool {
        self.queued
    }

    fn get_merged(&self) -> bool {
        self.merged
    }

    fn update_counter(&mut self, f: impl FnOnce(i32) -> i32) {
        self.counter = (f)(self.counter);
    }

    fn set_counter(&mut self, counter: i32) {
        self.counter = counter;
    }

    fn get_counter(&self) -> i32 {
        self.counter
    }
}

pub struct SharedPacked(AtomicU32);

impl SharedPacked {
    #[inline]
    pub fn load(&self, order: Ordering) -> Packed {
        Packed(self.0.load(order))
    }

    #[inline]
    pub fn compare_exchange(
        &self,
        current: Packed,
        new: Packed,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u32, u32> {
        self.0.compare_exchange(current.0, new.0, success, failure)
    }
}

pub const FLAG_MERGED: u32 = 1 << 31;
pub const FLAG_QUEUED: u32 = 1 << 30;

const VALUE_BITS: u32 = 30;
const VALUE_MASK: u32 = (1 << VALUE_BITS) - 1;
const VALUE_SIGN_BIT: u32 = 1 << (VALUE_BITS - 1);

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Packed(u32);

impl Packed {
    pub fn set_queued(&mut self, queued: bool) {
        let mask = FLAG_QUEUED;
        if queued {
            self.0 |= mask;
        } else {
            self.0 &= !mask
        }
    }

    pub fn set_merged(&mut self, merged: bool) {
        let mask = FLAG_MERGED;
        if merged {
            self.0 |= mask
        } else {
            self.0 &= !mask
        }
    }

    pub fn get_merged(&self) -> bool {
        self.is_merged()
    }

    pub fn get_queued(&self) -> bool {
        self.is_queued()
    }

    pub fn is_merged(&self) -> bool {
        self.0 & FLAG_MERGED != 0
    }

    pub fn is_queued(&self) -> bool {
        self.0 & FLAG_QUEUED != 0
    }

    // pub fn value(&self) -> i32 {
    //     (self.0 & VALUE_MASK) as _
    // }

    pub fn get_counter(&self) -> i32 {
        self.value()
    }

    // fn set_value(&mut self, value: i32) {
    //     assert!(value >= -(1 << 29) && value < (1 << 29));
    //     let v = (value as u32) & VALUE_MASK;
    //     // self.0 = (self.0 & !VALUE_MASK) | v;

    //     self.0 = v;
    // }

    fn value(&self) -> i32 {
        let raw = self.0 & VALUE_MASK;

        if raw & VALUE_SIGN_BIT != 0 {
            // Sign-extend from bit 29
            (raw | !VALUE_MASK) as i32
        } else {
            raw as i32
        }
    }

    pub fn set_counter(&mut self, value: i32) {
        self.set_value(value);
    }

    fn set_value(&mut self, value: i32) {
        assert!(value >= -(1 << 29) && value < (1 << 29));
        let v = (value as u32) & VALUE_MASK;
        self.0 = (self.0 & !VALUE_MASK) | v;
    }

    fn update_counter(&mut self, f: impl FnOnce(i32) -> i32) {
        self.set_value((f)(self.value()))
    }

    fn new_with(value: i32, merged: bool, queued: bool) -> Self {
        // Ensure value fits in signed 30 bits
        assert!(value >= -(1 << 29) && value < (1 << 29));

        let mut bits = (value as u32) & VALUE_MASK;

        if merged {
            bits |= FLAG_MERGED;
        }
        if queued {
            bits |= FLAG_QUEUED;
        }

        Self(bits)
    }

    fn new() -> Self {
        Self::new_with(0, false, false)
    }
}

#[test]
fn packed_vs_unpacked() {
    let mut packed = Packed::new();

    packed.set_merged(true);

    assert_eq!(packed.get_merged(), true);
    assert_eq!(packed.get_queued(), false);
    assert_eq!(packed.get_counter(), 0);

    packed.set_merged(false);
    assert_eq!(packed.get_merged(), false);
    assert_eq!(packed.get_queued(), false);
    assert_eq!(packed.get_counter(), 0);

    packed.set_value(100);
    assert_eq!(packed.get_merged(), false);
    assert_eq!(packed.get_queued(), false);
    assert_eq!(packed.get_counter(), 100);

    packed.set_value(-1);
    assert_eq!(packed.get_merged(), false);
    assert_eq!(packed.get_queued(), false);
    assert_eq!(packed.get_counter(), -1);

    packed.set_value(-100);
    packed.set_queued(true);
    assert_eq!(packed.get_merged(), false);
    assert_eq!(packed.get_queued(), true);
    assert_eq!(packed.get_counter(), -100);
}

impl SharedPacked {
    pub fn set_flag_queued(&self, queued: bool) {
        let mask = FLAG_QUEUED;
        if queued {
            self.0.fetch_or(mask, Ordering::Relaxed);
        } else {
            self.0.fetch_and(!mask, Ordering::Relaxed);
        }
    }

    pub fn set_flag_merged(&self, merged: bool) {
        let mask = FLAG_MERGED;
        if merged {
            self.0.fetch_or(mask, Ordering::Relaxed);
        } else {
            self.0.fetch_and(!mask, Ordering::Relaxed);
        }
    }

    fn new() -> Self {
        Self::new_with(0, false, false)
    }

    fn new_with(value: i32, merged: bool, queued: bool) -> Self {
        SharedPacked(AtomicU32::new(Packed::new_with(value, merged, queued).0))
    }

    // /// Get the signed 30-bit value (sign-extended)
    // fn value(self) -> i32 {
    //     let raw = self.0 & VALUE_MASK;

    //     if raw & VALUE_SIGN_BIT != 0 {
    //         // Sign-extend from bit 29
    //         (raw | !VALUE_MASK) as i32
    //     } else {
    //         raw as i32
    //     }
    // }

    // fn set_value(&mut self, value: i32) {
    //     assert!(value >= -(1 << 29) && value < (1 << 29));
    //     let v = (value as u32) & VALUE_MASK;
    //     self.0 = (self.0 & !VALUE_MASK) | v;
    // }

    pub fn is_merged(&self) -> bool {
        self.0.fetch_and(FLAG_MERGED, Ordering::Relaxed) != 0
    }

    pub fn is_queued(&self) -> bool {
        self.0.fetch_and(FLAG_QUEUED, Ordering::Relaxed) != 0
    }

    pub fn value(&self) -> u32 {
        self.0.fetch_and(VALUE_MASK, Ordering::Relaxed)
    }
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

            // Change this to be the shared packed version
            // shared: Atomic::new(Shared::new()),
            shared: SharedPacked::new(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum DecrementAction {
    DoNothing,
    Queue,
    Deallocate,
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

        if counter == u32::MAX {
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

            new.update_counter(|x| x + 1);

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

    pub fn decrement(&self) -> DecrementAction {
        let owner_tid = self.rcword.thread_id.get();
        let my_tid = ThreadId::current_thread();

        if owner_tid == Some(my_tid) {
            self.fast_decrement()
        } else {
            self.slow_decrement()
        }
    }

    pub fn fast_decrement(&self) -> DecrementAction {
        self.rcword.biased_counter.update(|x| x - 1);
        if self.rcword.biased_counter.get() > 0 {
            return DecrementAction::DoNothing;
        }

        let mut new;

        loop {
            let old = self.rcword.shared.load(Ordering::Relaxed);
            new = old;
            new.set_merged(true);
            if self
                .rcword
                .shared
                .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        if new.get_counter() == 0 {
            DecrementAction::Deallocate
        } else {
            self.rcword.thread_id.set(None);

            DecrementAction::DoNothing
        }
    }

    pub fn slow_decrement(&self) -> DecrementAction {
        let mut old;
        let mut new;
        loop {
            old = self.rcword.shared.load(Ordering::Relaxed);
            new = old;

            new.update_counter(|x| x - 1);

            println!("Slow decrement: {:?}", new);

            if new.get_counter() < 0 {
                new.set_queued(true);
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

        if old.get_queued() != new.get_queued() {
            DecrementAction::Queue
        } else if new.get_merged() && new.get_counter() == 0 {
            DecrementAction::Deallocate
        } else {
            DecrementAction::Deallocate
        }
    }

    // TODO: Figure out how to do this right?
    fn has_unique_ref(&self) -> bool {
        let word = self.rcword.shared.load(Ordering::Acquire);
        let mut count = word.get_counter();

        if word.get_counter() == 0 {
            let owner = self.rcword.thread_id.get();
            match owner {
                None => {}
                Some(tid) if tid == ThreadId::current_thread() => {
                    count = self.rcword.biased_counter.get() as i32;
                }
                Some(_) => {
                    count = 2;
                }
            }
        }

        count == 1
    }
}

pub struct Wrapper(Box<ManuallyDrop<dyn BiasedMerge>>);

unsafe impl Send for Wrapper {}
unsafe impl Sync for Wrapper {}

#[derive(Default)]
pub struct TypeMap {
    inner: Vec<Wrapper>,
    map: HashMap<Option<ThreadId>, Vec<Wrapper>>,
}

pub trait BiasedMerge {
    fn merge(self);
    fn meta_outer(&self) -> &RcWord;
    unsafe fn drop_contents_and_maybe_box_outer(&mut self);
}

impl<T: ?Sized> BiasedMerge for BiasedRc<T> {
    fn merge(mut self) {
        let mut old;
        let mut new;
        loop {
            old = self.meta().shared.load(Ordering::Acquire);
            new = old;
            new.update_counter(|x| x + self.meta().biased_counter.get() as i32);
            new.set_merged(true);

            if self
                .meta()
                .shared
                .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }

            if new.get_counter() == 0 {
                unsafe { self.drop_contents_and_maybe_box() };
            } else {
                self.meta().thread_id.set(None);
            }
        }

        std::mem::forget(self);
    }

    fn meta_outer(&self) -> &RcWord {
        self.meta()
    }

    unsafe fn drop_contents_and_maybe_box_outer(&mut self) {
        unsafe { self.drop_contents_and_maybe_box() }
    }
}

pub static QUEUE: LazyLock<Mutex<TypeMap>> = LazyLock::new(|| Mutex::new(TypeMap::default()));

impl TypeMap {
    pub fn insert<T: ?Sized + 'static>(&mut self, value: BiasedRc<T>) {
        self.inner.push(Wrapper(Box::new(ManuallyDrop::new(value))));
    }

    pub fn enqueue<T: ?Sized + 'static>(value: &BiasedRc<T>) {
        let key = value.meta().thread_id.get();
        let mut guard = QUEUE.lock().unwrap();

        if let Some(q) = guard.map.get_mut(&key) {
            q.push(Wrapper(Box::new(ManuallyDrop::new(BiasedRc::from_inner(
                value.ptr,
            )))));
        } else {
            guard.map.insert(
                key,
                vec![Wrapper(Box::new(ManuallyDrop::new(BiasedRc::from_inner(
                    value.ptr,
                ))))],
            );
        }
    }

    pub fn run_explicit_merge() {
        QUEUE
            .lock()
            .unwrap()
            .map
            .get_mut(&Some(ThreadId::current_thread()))
            .map(Self::explicit_merge);
    }

    pub fn explicit_merge(values: &mut Vec<Wrapper>) {
        println!(
            "Running explicit merge on thread: {:?} with count: {}",
            std::thread::current().id(),
            values.len()
        );

        for value in values.drain(..) {
            let mut value = value.0;
            let mut old;
            let mut new;
            loop {
                old = value.meta_outer().shared.load(Ordering::Acquire);
                new = old;
                new.update_counter(|x| x + value.meta_outer().biased_counter.get() as i32);
                new.set_merged(true);

                if value
                    .meta_outer()
                    .shared
                    .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }

            if new.get_counter() == 0 {
                println!("invoking the destructor");
                unsafe { value.drop_contents_and_maybe_box_outer() };
            } else {
                value.meta_outer().thread_id.set(None);
            }

            drop(value);
        }
    }
}

impl BiasedRc<dyn Any> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<BiasedRc<T>, Self> {
        if (*self).is::<T>() {
            let ptr = self.ptr.cast::<RcBox<T>>();
            mem::forget(self);
            Ok(BiasedRc::from_inner(ptr))
        } else {
            Err(self)
        }
    }
}

impl BiasedRc<dyn Any + Sync + Send> {
    #[inline]
    pub fn downcast<T: Any + Sync + Send>(self) -> Result<BiasedRc<T>, Self> {
        if (*self).is::<T>() {
            let ptr = self.ptr.cast::<RcBox<T>>();
            mem::forget(self);
            Ok(BiasedRc::from_inner(ptr))
        } else {
            Err(self)
        }
    }
}

impl<T: Any + 'static> From<BiasedRc<T>> for BiasedRc<dyn Any + 'static> {
    #[inline]
    fn from(src: BiasedRc<T>) -> Self {
        let ptr = src.ptr.as_ptr() as *mut RcBox<dyn Any>;
        mem::forget(src);
        Self::from_inner(unsafe { NonNull::new_unchecked(ptr) })
    }
}

impl<T: Any + Sync + Send + 'static> From<BiasedRc<T>>
    for BiasedRc<dyn Any + Sync + Send + 'static>
{
    #[inline]
    fn from(src: BiasedRc<T>) -> Self {
        let ptr = src.ptr.as_ptr() as *mut RcBox<dyn Any + Sync + Send>;
        mem::forget(src);
        Self::from_inner(unsafe { NonNull::new_unchecked(ptr) })
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

pub struct BiasedRc<T: ?Sized + 'static> {
    ptr: NonNull<RcBox<T>>,
    phantom2: PhantomData<T>,
}

impl<T: ?Sized> BiasedRc<T> {
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

    #[must_use]
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.get_box().has_unique_ref() {
            unsafe { Some(Self::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr = this.ptr.as_ptr();

        // Safety: Neccessary for `from_raw()` (when implemented), retains provenance.
        // Besides that, does basically the same thing as `data()` or `get_mut_unchecked()`.
        unsafe { ptr::addr_of_mut!((*ptr).data) }
    }

    #[must_use = "Memory will leak if the result is not used"]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = Self::as_ptr(&this);
        mem::forget(this);
        ptr
    }

    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // Safety: covered by the safety contract for this function
        let box_ptr = unsafe { RcBox::<T>::ptr_from_data_ptr(ptr) };

        Self::from_inner(NonNull::new(box_ptr as *mut _).expect("invalid pointer"))
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &BiasedRc<T>) -> bool {
        std::ptr::eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    #[inline]
    pub fn ptr_eq_pin(this: &Pin<Self>, other: &Pin<BiasedRc<T>>) -> bool {
        // SAFETY: we are not moving anything and we don't expose any pointers.
        let this = unsafe { Self::pin_get_ref(this) };
        let other = unsafe { BiasedRc::<T>::pin_get_ref(other) };
        std::ptr::eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        // let meta = this.meta();
        // meta.shared.load(Ordering::SeqCst).counter as usize + meta.biased_counter.get() - 1

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
    }
}

impl<T> BiasedRc<T> {
    #[inline]
    pub fn new(data: T) -> Self {
        let mut inner = RcBox::allocate(Self::build_new_meta());
        let inner = unsafe { inner.as_mut() };
        inner.data.write(data);
        Self::from_inner(unsafe { inner.assume_init() }.into())
    }

    /// Creates a new `HybridRc` with uninitialized contents.
    #[inline]
    pub fn new_uninit() -> BiasedRc<mem::MaybeUninit<T>> {
        let inner = RcBox::allocate(Self::build_new_meta());
        BiasedRc::from_inner(inner)
    }

    /// Creates a new `HybridRc` with uninitialized contents, with the memory being filled with
    /// 0 bytes.
    ///
    /// See [`MaybeUninit::zeroed()`] for examples of correct and incorrect usage of this method.
    ///
    /// [`MaybeUninit::zeroed()`]: mem::MaybeUninit::zeroed
    #[inline]
    pub fn new_zeroed() -> BiasedRc<mem::MaybeUninit<T>> {
        let mut inner = RcBox::allocate(Self::build_new_meta());
        unsafe { inner.as_mut() }.data = mem::MaybeUninit::zeroed();
        BiasedRc::from_inner(inner)
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

    pub fn try_new_uninit() -> Result<BiasedRc<mem::MaybeUninit<T>>, AllocError> {
        let inner = RcBox::try_allocate(Self::build_new_meta()).map_err(|_| AllocError)?;
        Ok(BiasedRc::from_inner(inner.into()))
    }

    pub fn try_new_zeroed() -> Result<BiasedRc<mem::MaybeUninit<T>>, AllocError> {
        let mut inner = RcBox::try_allocate(Self::build_new_meta()).map_err(|_| AllocError)?;
        unsafe { inner.as_mut() }.data = mem::MaybeUninit::zeroed();
        Ok(BiasedRc::from_inner(inner))
    }

    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let owner = this.meta().thread_id.get();
        match owner {
            None => Self::try_unwrap_internal(this),
            Some(tid) if tid == ThreadId::current_thread() => {
                let local_count = this.meta().biased_counter.get();

                if local_count == 1 {
                    Self::try_unwrap_internal_same_thread(this)
                } else {
                    Err(this)
                }
            }
            // Has an owner on a different thread.
            Some(_) => Err(this),
        }
    }

    fn try_unwrap_internal_same_thread(this: Self) -> Result<T, Self> {
        let meta = this.meta();
        let old = meta.shared.load(Ordering::Relaxed);
        if old.get_counter() != 0 {
            Err(this)
        } else {
            meta.thread_id.set(None);
            let copy = unsafe { ptr::read(Self::as_ptr(&this)) };

            // Deallocate the box?
            unsafe { RcBox::dealloc(this.ptr) };

            mem::forget(this);

            Ok(copy)
        }
    }

    /// Returns the inner value, if this `HybridRc` is the only strong reference to it, assuming
    /// that there are no (other) local references to the value.
    ///
    /// Used internally by `try_unwrap()`.
    #[inline]
    fn try_unwrap_internal(this: Self) -> Result<T, Self> {
        let meta = this.meta();
        let mut new;
        let mut old;

        // loop {
        old = meta.shared.load(Ordering::Relaxed);
        new = old;

        old.set_counter(1);
        new.set_counter(0);

        if meta
            .shared
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            Err(this)
        } else {
            meta.thread_id.set(None);
            let copy = unsafe { ptr::read(Self::as_ptr(&this)) };

            // Deallocate the box?
            unsafe { RcBox::dealloc(this.ptr) };

            mem::forget(this);

            Ok(copy)
        }
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

impl<T: ?Sized> Deref for BiasedRc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.data()
    }
}

impl<T: ?Sized> Borrow<T> for BiasedRc<T> {
    #[inline]
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> AsRef<T> for BiasedRc<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> Clone for BiasedRc<T> {
    #[inline]
    fn clone(&self) -> Self {
        self.get_box().increment();
        Self::from_inner(self.ptr)
    }
}

impl<T: ?Sized + 'static> Drop for BiasedRc<T> {
    #[inline]
    fn drop(&mut self) {
        match self.get_box().decrement() {
            DecrementAction::DoNothing => {}
            DecrementAction::Queue => {
                // Enqueue the value
                TypeMap::enqueue(self);
            }
            DecrementAction::Deallocate => {
                unsafe { self.drop_contents_and_maybe_box() };
            }
        }
    }
}

// Propagate some useful traits implemented by the inner type

impl<T: Default> Default for BiasedRc<T> {
    /// Creates a new `HybridRc`, with the `Default` value for `T`.
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T: ?Sized + PartialEq> PartialEq<BiasedRc<T>> for BiasedRc<T> {
    /// Equality for `HybridRc`s.
    ///
    /// Two `HybridRc`s are equal if their inner values are equal, independent of if they are
    /// stored in the same allocation.
    #[inline]
    fn eq(&self, other: &BiasedRc<T>) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq> Eq for BiasedRc<T> {}

impl<T: ?Sized + Hash> Hash for BiasedRc<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Self::data(self).hash(state);
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd<BiasedRc<T>> for BiasedRc<T> {
    /// Partial comparison for `HybridRc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    #[inline]
    fn partial_cmp(&self, other: &BiasedRc<T>) -> Option<cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for BiasedRc<T> {
    /// Comparison for `HybridRc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for BiasedRc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&Self::data(self), f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for BiasedRc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&Self::data(self), f)
    }
}

// `HybridRc` can be formatted as a pointer.
impl<T: ?Sized> fmt::Pointer for BiasedRc<T> {
    /// Formats the value using the given formatter.
    ///
    /// If the `#` flag is used, the state (shared/local) is written after the address.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Self::as_ptr(self), f)
    }
}

/// `HybridRc<T>` is always `Unpin` itself, because the data value is on the heap,
/// so moving `HybridRc<T>` doesn't move the content even if `T` is not `Unpin`.
///
/// This allows unpinning e.g. `Pin<Box<HybridRc<T>>>` but not any `Pin<HybridRc<T>>`!
impl<T: ?Sized> Unpin for BiasedRc<T> {}

unsafe impl<T: ?Sized + Sync + Send> Send for BiasedRc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for BiasedRc<T> {}

#[test]
fn does_this_work() {
    let value = BiasedRc::new(10);
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
    let value = BiasedRc::new(Foo { foo: 10 });

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
    let value = BiasedRc::new(Foo { foo: 10 });
    let cloned = BiasedRc::clone(&value);

    drop(value);

    println!("Now we're done");

    drop(cloned);
}

#[test]
fn test_queue_impl() {
    struct Foo {
        foo: String,
    }
    impl Drop for Foo {
        fn drop(&mut self) {
            println!("Calling drop: {}", self.foo);
        }
    }
    let value = BiasedRc::new(Foo {
        foo: "hello world".to_string(),
    });
    let cloned = BiasedRc::clone(&value);

    let thread = std::thread::spawn(move || {
        drop(cloned);

        // Run explicit merge:
        TypeMap::run_explicit_merge();
    });

    thread.join().unwrap();

    TypeMap::run_explicit_merge();
}

#[test]
fn test_try_unwrap_impl_same_thread() {
    #[derive(Debug)]
    struct Foo {
        foo: String,
    }
    impl Drop for Foo {
        fn drop(&mut self) {
            println!("Calling drop: {}", self.foo);
        }
    }
    let value = BiasedRc::new(Foo {
        foo: "hello world".to_string(),
    });
    let cloned = BiasedRc::clone(&value);

    let failed_unwrap = BiasedRc::try_unwrap(value).unwrap_err();

    drop(failed_unwrap);

    assert!(BiasedRc::try_unwrap(cloned).is_ok());

    // let thread = std::thread::spawn(move || {
    //     drop(cloned);

    //     // Run explicit merge:
    //     TypeMap::run_explicit_merge();
    // });

    // thread.join().unwrap();

    // TypeMap::run_explicit_merge();
}

#[test]
fn test_try_unwrap_impl_moved_thread() {
    #[derive(Debug)]
    struct Foo {
        foo: String,
    }
    impl Drop for Foo {
        fn drop(&mut self) {
            println!("Calling drop: {}", self.foo);
        }
    }
    let value = BiasedRc::new(Foo {
        foo: "hello world".to_string(),
    });
    let cloned = BiasedRc::clone(&value);

    let failed_unwrap = BiasedRc::try_unwrap(value).unwrap_err();

    drop(failed_unwrap);

    assert!(BiasedRc::try_unwrap(cloned).is_ok());

    // let thread = std::thread::spawn(move || {
    //     drop(cloned);

    //     // Run explicit merge:
    //     TypeMap::run_explicit_merge();
    // });

    // thread.join().unwrap();

    // TypeMap::run_explicit_merge();
}
